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

/// What a step does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// Text typed into the panel.
    Prompt { text: String },
    /// The host node's image input, rendered at the playhead when the run
    /// starts. Leaves the comp as display-referred sRGB (see `image`).
    Reference,
    /// The host node's TEXT input: whatever Prompt node is wired into it,
    /// read when the run starts. Keyed exactly like a Prompt step with the
    /// same text, so moving a prompt out into the graph re-bills nothing.
    PromptIn,
    /// The host node's MATTE input at the playhead — white where a fill or
    /// an edit should land.
    Matte,
    /// The host node's image input over a frame RANGE from the playhead,
    /// encoded as an mp4 when the run starts: what a video-to-video model
    /// reads. Leaves the comp as display-referred sRGB, like `Reference`.
    ClipIn {
        /// Length of the clip, in seconds of comp time.
        #[serde(default = "default_clip_seconds")]
        seconds: f32,
    },
    /// The host node's MATTE input over the same range as the pipeline's
    /// `ClipIn` (`Pipeline::clip_seconds`): white where a video model
    /// should work (an object to remove, say).
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
            Op::Prompt { .. } | Op::PromptIn | Op::Reference | Op::Matte | Op::ClipIn { .. } | Op::MatteClip => {
                vec![]
            }
            Op::Output => vec![(Pipe::Image, "image")],
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
            Op::Prompt { .. } | Op::PromptIn => Some(Pipe::Text),
            Op::Output => None,
            Op::ClipIn { .. } | Op::MatteClip => Some(Pipe::Video),
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
        match self.ports().get(port) {
            Some(&(want, _)) => {
                want == pipe || (matches!(self, Op::Output) && pipe == Pipe::Video)
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
            Op::PromptIn => "Prompt In",
            Op::Reference => "Input",
            Op::Matte => "Matte",
            Op::ClipIn { .. } => "Input Clip",
            Op::MatteClip => "Matte Clip",
            Op::Generate { model, .. } => catalog::get(model).map(|m| m.label).unwrap_or("Unknown Model"),
            Op::Comfy { .. } => "ComfyUI",
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
            Op::PromptIn => "prompt in".into(),
            Op::Reference => "input".into(),
            Op::Matte => "matte".into(),
            Op::ClipIn { seconds } => format!("clip {seconds:.1}s"),
            Op::MatteClip => "matte clip".into(),
            Op::Generate { model, .. } => match catalog::get(model) {
                Some(m) => m.short.into(),
                None => "unknown".into(),
            },
            Op::Comfy { .. } => "comfy".into(),
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
}

/// The extensions a video step's result can come back as.
pub const VIDEO_EXTS: &[&str] = &["mp4", "mov", "webm"];

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
}

impl Default for Pipeline {
    /// Prompt -> text-to-image -> Output: the shortest thing that makes a
    /// picture.
    fn default() -> Self {
        let mut p = Pipeline { steps: vec![], takes: vec![], take: 0 };
        let prompt = p.push(Op::Prompt { text: String::new() }, [-0.55, -0.45]);
        let gen = p.push(
            Op::Generate { model: catalog::DEFAULT_MODEL.into(), seed: 1, opts: Opts::default() },
            [0.0, 0.0],
        );
        let out = p.push(Op::Output, [0.0, 0.55]);
        p.connect(prompt, gen, 0);
        p.connect(gen, out, 0);
        p
    }
}

impl Pipeline {
    /// The fixed pipeline of a one-model node: the model's inputs, each fed
    /// by the source its port names (a typed Prompt, the node's image, the
    /// node's matte), into the model, into Output. `None` for an id this
    /// build does not know.
    pub fn for_model(id: &str) -> Option<Self> {
        let m = catalog::get(id)?;
        let mut p = Pipeline { steps: vec![], takes: vec![], take: 0 };
        let gen = p.push(Op::Generate { model: m.id.into(), seed: 1, opts: Opts::default() }, [0.0, 0.0]);
        let ports = task_ports(m.task);
        let n = ports.len().max(1) as f32;
        for (port, (pipe, name)) in ports.into_iter().enumerate() {
            let op = match (pipe, name) {
                (Pipe::Text, _) => Op::Prompt { text: String::new() },
                (Pipe::Video, "mask") => Op::MatteClip,
                (Pipe::Video, _) => Op::ClipIn { seconds: default_clip_seconds() },
                (_, "mask") => Op::Matte,
                _ => Op::Reference,
            };
            let x = -0.6 + 1.2 * (port as f32 + 0.5) / n;
            let src = p.push(op, [x, -0.55]);
            p.connect(src, gen, port);
        }
        let out = p.push(Op::Output, [0.0, 0.55]);
        p.connect(gen, out, 0);
        Some(p)
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
    pub fn remove(&mut self, i: usize) -> bool {
        if i >= self.steps.len() || matches!(self.steps[i].op, Op::Output) {
            return false;
        }
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
        true
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
        let ports = step.op.ports();
        for (k, (_, name)) in ports.iter().enumerate() {
            match step.inputs.get(k).copied().flatten() {
                Some(src) => self.visit(src, state, order)?,
                // Comfy's ports are all optional: a workflow uses the ones
                // it has placeholders for.
                None if matches!(step.op, Op::Comfy { .. }) => {}
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

    /// Does the pipeline read the host's image input? Its matte input?
    pub fn wants_reference(&self) -> bool {
        self.steps.iter().any(|s| matches!(s.op, Op::Reference))
    }
    pub fn wants_matte(&self) -> bool {
        self.steps.iter().any(|s| matches!(s.op, Op::Matte))
    }
    pub fn wants_prompt(&self) -> bool {
        self.steps.iter().any(|s| matches!(s.op, Op::PromptIn))
    }
    /// Does it read the host's image input as a clip? Its matte as one?
    pub fn wants_clip(&self) -> bool {
        self.steps.iter().any(|s| matches!(s.op, Op::ClipIn { .. }))
    }
    pub fn wants_matte_clip(&self) -> bool {
        self.steps.iter().any(|s| matches!(s.op, Op::MatteClip))
    }
    /// How long a run's clips are: the first `ClipIn`'s length (a
    /// `MatteClip` follows it, so the two always line up frame for frame),
    /// else the default.
    pub fn clip_seconds(&self) -> f32 {
        self.steps
            .iter()
            .find_map(|s| match s.op {
                Op::ClipIn { seconds } => Some(seconds),
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
        for m in catalog::CATALOG {
            let p = Pipeline::for_model(m.id).unwrap();
            let out = p.output().unwrap();
            let plan = p.plan(out).unwrap_or_else(|e| panic!("{}: {e}", m.id));
            let g = p.model_step().unwrap();
            assert!(p.steps[g].inputs.iter().all(|i| i.is_some()), "{}", m.id);
            assert_eq!(plan.len(), p.steps.len(), "{}: every step is on the way out", m.id);
            let ports = task_ports(m.task);
            assert_eq!(p.wants_matte(), ports.iter().any(|&(k, n)| k == Pipe::Image && n == "mask"), "{}", m.id);
            assert_eq!(p.wants_matte_clip(), ports.iter().any(|&(k, n)| k == Pipe::Video && n == "mask"), "{}", m.id);
            assert_eq!(p.wants_clip(), ports.iter().any(|&(k, n)| k == Pipe::Video && n != "mask"), "{}", m.id);
            assert_eq!(p.wants_reference(), ports.iter().any(|&(k, n)| k == Pipe::Image && n != "mask"), "{}", m.id);
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
    fn clip_steps_feed_video_ports_and_old_files_still_load() {
        let mut p = Pipeline { steps: vec![], takes: vec![], take: 0 };
        let c = p.push(Op::ClipIn { seconds: 3.0 }, [0.0, 0.0]);
        let r = p.push(Op::Reference, [0.0, 0.0]);
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
}
