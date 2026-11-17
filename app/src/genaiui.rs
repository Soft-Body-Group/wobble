//! The app half of the GenAI node: its step graph drawn INSIDE the node,
//! its panel, and a run from the click to a take on disk.
//!
//! The `genai` crate owns what a pipeline is and how it runs; this module
//! decides WHEN, and what the user sees on the way. A run is a small state
//! machine, one at a time across the app:
//!
//! 1. **Capture** — the node's inputs are rendered at the playhead through
//!    the GraphBake loop (a one-frame `BakeSink::Capture`), read back, taken
//!    to Rec.709, shrunk to `REF_MP` and encoded as sRGB PNG. No second
//!    render path: a capture IS a one-frame bake of the input.
//! 2. **Plan + quote** — `genai::run::plan` keys every step and finds what
//!    is cached; if anything is billed, the price is asked of the service on
//!    a thread. Nothing has been sent yet.
//! 3. **Confirm** — the panel shows the price (and the balance, on
//!    credits). Only a click on "pay & run" spends anything, and the relay
//!    refuses a job priced above what was shown. A run with nothing billed
//!    (all cached, or ComfyUI only) goes straight through.
//! 4. **Run** — `genai::run::start` on its own thread; events are drained
//!    once a redraw. The take lands as an UNDO STEP (one generation is one
//!    user action — unlike a tracker's per-frame poll) and the node loads it
//!    into its reader slot.
//!
//! Keys, the route (credits or your own key) and the ComfyUI address are
//! machine state (`genai::config`): never in a setup, never in prefs.

use crate::{innards, theme, Gfx};
use egui::{Color32, Pos2, Stroke, Ui, Vec2};
use engine::graph::{GenAiParams, NodeKind};
use genai::catalog::{self, Task};
use genai::config::{self, Config, Origin, Route};
use genai::pipeline::{Op, Opts, Pipeline};
use genai::run::{self, Clip, Event, Inputs, Plan, Png, StepState};
use genai::service::{self, Quote};
use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;

/// Panel footprint: two tiles wide (steps | run), two tall.
pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;
/// Largest reference we send, in megapixels (see `genai::image::shrink`).
pub const REF_MP: f32 = 2.0;
/// A clip's short side, at most, in pixels: what the video models we call
/// work at, and the floor Kling asks for.
pub const CLIP_SHORT: u32 = 720;

/// Capture "ports" beyond the node's own inputs: input 0 (the image) and
/// input 1 (the matte) rendered over the run's RANGE for a `ClipIn` /
/// `MatteClip` step, rather than at the playhead. `port - CLIP` is the
/// input rendered.
pub const CLIP: usize = 10;
pub const MATTE_CLIP: usize = 11;

/// The node input a capture port renders.
fn input_of_port(port: usize) -> usize {
    if port >= CLIP {
        port - CLIP
    } else {
        port
    }
}

/// Can this build send a clip? It must encode an mp4 (a release build's
/// FFmpeg decodes only) and play the video take back.
pub fn clip_ok() -> bool {
    #[cfg(feature = "movie")]
    {
        video_ok() && engine::movie::can_encode_mp4()
    }
    #[cfg(not(feature = "movie"))]
    {
        false
    }
}

const NO_CLIP: &str = "this build cannot send a clip: it needs --features app/movie and an FFmpeg with an encoder";

/// Can this build play a video take? Only a build that decodes movies
/// (`--features app/movie`) offers a video step: one it cannot play back
/// would bill for a clip and show nothing (the `SUPPORTED_EXTS` rule).
pub fn video_ok() -> bool {
    engine::decode::is_supported(std::path::Path::new("take.mp4"))
}

/// Is this model one this build may offer?
fn offered(m: &catalog::RemoteModel) -> bool {
    if m.task.takes_clip() {
        clip_ok()
    } else {
        !m.task.is_video() || video_ok()
    }
}

/// The models this build adds as one-model nodes (`editor::node_catalog`).
pub fn offered_models() -> impl Iterator<Item = &'static catalog::RemoteModel> {
    catalog::CATALOG.iter().filter(|m| offered(m))
}

const NO_VIDEO: &str = "this build cannot play a video take: build with --features app/movie";

fn params(kind: &NodeKind) -> Option<&GenAiParams> {
    match kind {
        NodeKind::GenAi(p) => Some(p),
        _ => None,
    }
}

fn params_mut(kind: &mut NodeKind) -> Option<&mut GenAiParams> {
    match kind {
        NodeKind::GenAi(p) => Some(p),
        _ => None,
    }
}

// ------------------------------------------------------------ the dive

/// A GenAI Pipeline's steps are opened like a Noise's field graph: Enter
/// on the node dives into an ordinary node graph whose nodes are
/// `NodeKind::GenAiStep`s, one per step, wired like any other graph. The
/// steps live on the owner as a `genai::Pipeline`; these two functions are
/// the round trip (the Particles/SDF pattern). Step index + 1 is the node
/// id going in; coming out, steps are re-indexed by id order.
pub fn to_editor(pl: &Pipeline) -> crate::editor::EditorGraph {
    let mut g = crate::editor::EditorGraph::default();
    // Pipelines drawn inside the node face (before the dive) kept their
    // positions in the face's [-1, 1] square; spread those out.
    let face = pl.steps.iter().all(|s| s.at[0].abs() <= 1.0 && s.at[1].abs() <= 1.0);
    let k = if face { 260.0 } else { 1.0 };
    for (i, s) in pl.steps.iter().enumerate() {
        let id = i as u64 + 1;
        let pos = egui::pos2(s.at[0] * k, s.at[1] * k);
        g.nodes.insert(id, crate::editor::ENode::new(NodeKind::GenAiStep(s.op.clone()), pos));
    }
    for (to, s) in pl.steps.iter().enumerate() {
        for (port, src) in s.inputs.iter().enumerate() {
            if let Some(from) = src {
                g.edges.push((to as u64 + 1, port, *from as u64 + 1));
            }
        }
    }
    g.next_id = pl.steps.len() as u64 + 1;
    g
}

/// The other direction: the dive's graph back into steps. Anything that is
/// not a step is dropped, and so is a wire into a port the step lacks or of
/// the wrong pipe (`Op::accepts`, the pipeline's one wiring rule).
pub fn from_editor(g: &crate::editor::EditorGraph) -> Vec<genai::Step> {
    let ids: Vec<u64> =
        g.nodes.iter().filter(|(_, n)| matches!(n.kind, NodeKind::GenAiStep(_))).map(|(id, _)| *id).collect();
    let index = |id: u64| ids.iter().position(|&x| x == id);
    let mut steps: Vec<genai::Step> = ids
        .iter()
        .map(|id| {
            let n = &g.nodes[id];
            let NodeKind::GenAiStep(op) = &n.kind else { unreachable!() };
            genai::Step { op: op.clone(), at: [n.pos.x, n.pos.y], inputs: vec![None; op.ports().len()] }
        })
        .collect();
    for &(dst, port, src) in &g.edges {
        let (Some(to), Some(from)) = (index(dst), index(src)) else { continue };
        let Some(pipe) = steps[from].op.output() else { continue };
        if steps[to].op.accepts(port, pipe) {
            steps[to].inputs[port] = Some(from);
        }
    }
    steps
}

/// The step nodes the dive's add menu offers: the node's inputs, one node
/// per model (no model menu, the same rule as the comp), ComfyUI, Output.
pub fn step_entries() -> Vec<(&'static str, Op)> {
    let mut v = vec![Op::Prompt { text: String::new() }, Op::PromptIn, Op::Reference, Op::Matte];
    if clip_ok() {
        v.extend([Op::ClipIn { seconds: 5.0 }, Op::MatteClip]);
    }
    v.extend(offered_models().map(|m| Op::Generate { model: m.id.into(), seed: 1, opts: Opts::default() }));
    v.push(Op::Comfy { workflow: String::new(), seed: 1, commercial: false });
    v.push(Op::Output);
    v.into_iter().map(|op| (op.name(), op)).collect()
}

/// A step node's properties, inside the dive. `v` is the OWNER's view.
pub fn step_panel(ui: &mut Ui, node: u64, op: &mut Op, v: &View) {
    let id = move |what: &str| egui::Id::new(("genai-step", node, what.to_string()));
    let _ = op_fields(ui, &id, op, v, false);
    ui.add_space(4.0);
    ui.weak(small("run… is on the pipeline node: step out (U) to run"));
}

// ------------------------------------------------------------ graph marks

/// A four-point sparkle over the node's disc.
pub fn glyph(painter: &egui::Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = theme::node_ink();
    let (tip, waist) = (r * 0.72, r * 0.2);
    for k in 0..4 {
        let a = k as f32 * std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_2;
        let d = Vec2::angled(a);
        let s = Vec2::angled(a + std::f32::consts::FRAC_PI_4) * waist;
        let s2 = Vec2::angled(a - std::f32::consts::FRAC_PI_4) * waist;
        // Each arm is a convex kite; four of them make the star.
        painter.add(egui::Shape::convex_polygon(vec![c, c + s2, c + d * tip, c + s], ink, Stroke::NONE));
    }
}

/// Does running this node's pipeline spend money? (A model step on a
/// remote service. Cached steps still count: whether they are cached is a
/// hash of the reference, which the graph does not render every frame.)
pub fn spends(p: &GenAiParams) -> bool {
    p.pipeline.steps.iter().any(|s| s.op.is_remote())
}

/// The money badge: a `$` off the node's lower-left corner (a GenAI node
/// never declares weights, so the weights badge never wants that corner).
/// Manifesto 8's rule, for a new currency: a node that bills when you press
/// it says so where you can see it.
pub fn money_badge(painter: &egui::Painter, sp: Pos2, r: f32) {
    let br = (r * 0.4).clamp(3.5, 6.0);
    let c = sp + Vec2::new(-r * 0.95 - 4.0, r * 0.95 + 4.0);
    painter.circle(c, br, theme::cost_money(), Stroke::new((br * 0.25).max(1.0), theme::node_outline()));
    if br >= 4.5 {
        painter.text(
            c,
            egui::Align2::CENTER_CENTER,
            "$",
            egui::FontId::proportional(br * 1.5),
            theme::node_outline(),
        );
    }
}

// ------------------------------------------------------------ run state

/// Every GenAI run in the app. One at a time: a second click while one is
/// going says so rather than queueing money.
pub struct Runs {
    pub cfg: Config,
    pub run: Option<Run>,
    /// Captures handed over by the bake loop: (node, port, working-space
    /// RGBA, w, h). Written from inside `step_graph_bake`, where only field
    /// borrows are available, and consumed at `finish_graph_bake`.
    pub captured: Vec<(u64, usize, Vec<f32>, u32, u32)>,
    /// A clip capture's frames as they come off the bake loop, already
    /// Rec.709, shrunk to `CLIP_SHORT` and 8-bit sRGB RGB (a 2K float
    /// frame kept per tick would be gigabytes): (node, port, rgb, w, h).
    pub clip_frames: Vec<(u64, usize, Vec<u8>, u32, u32)>,
    account: Account,
}

impl Runs {
    /// The bake loop's hand-off for one captured frame (field borrows only:
    /// it runs inside `step_graph_bake`). A still is kept as float for the
    /// PNG; a clip frame is converted on the spot.
    pub fn take_capture(&mut self, working: engine::color::Colorspace, node: u64, port: usize, mut px: Vec<f32>, w: u32, h: u32) {
        if port < CLIP {
            self.captured.push((node, port, px, w, h));
            return;
        }
        if port == CLIP {
            to_rec709(&mut px, working);
        } else {
            // Coverage rides every channel of a matte; take red, as grey.
            for p in px.chunks_exact_mut(4) {
                p[1] = p[0];
                p[2] = p[0];
            }
        }
        let (rgb, cw, ch) = clip_frame(&px, w, h, port == CLIP);
        self.clip_frames.push((node, port, rgb, cw, ch));
    }
}

/// One clip frame: box-averaged down until its short side is at most
/// `CLIP_SHORT` (exactly, not by a whole factor: a 2K plate must come out
/// at 720, not 540, or Kling refuses it), sides even (4:2:0 needs them),
/// encoded to 8-bit RGB — sRGB for a picture, straight coverage for a matte.
fn clip_frame(rgba: &[f32], w: u32, h: u32, picture: bool) -> (Vec<u8>, u32, u32) {
    let scale = (CLIP_SHORT as f64 / w.min(h).max(1) as f64).min(1.0);
    let even = |v: f64| ((v / 2.0).floor() as u32).max(1) * 2;
    let (ow, oh) = (even(w as f64 * scale).min(w & !1).max(2), even(h as f64 * scale).min(h & !1).max(2));
    let (sx, sy) = (w as f64 / ow as f64, h as f64 / oh as f64);
    let mut out = Vec::with_capacity((ow * oh * 3) as usize);
    for y in 0..oh {
        let (y0, y1) = ((y as f64 * sy) as u32, (((y + 1) as f64 * sy).ceil() as u32).clamp(1, h));
        for x in 0..ow {
            let (x0, x1) = ((x as f64 * sx) as u32, (((x + 1) as f64 * sx).ceil() as u32).clamp(1, w));
            let mut acc = [0f32; 3];
            let mut n = 0f32;
            for yy in y0..y1.max(y0 + 1).min(h) {
                for xx in x0..x1.max(x0 + 1).min(w) {
                    let k = ((yy * w + xx) * 4) as usize;
                    for c in 0..3 {
                        acc[c] += rgba[k + c];
                    }
                    n += 1.0;
                }
            }
            for a in acc {
                let v = (a / n.max(1.0)).clamp(0.0, 1.0);
                let v = if picture { genai::image::srgb_encode(v) } else { v };
                out.push((v * 255.0 + 0.5) as u8);
            }
        }
    }
    (out, ow, oh)
}

#[cfg_attr(not(feature = "commerce"), allow(dead_code))]
enum Account {
    Unknown,
    Fetching(Receiver<Result<(String, f64), String>>),
    Known(Result<(String, f64), String>),
}

impl Default for Runs {
    fn default() -> Self {
        Self { cfg: Config::load(), run: None, captured: vec![], clip_frames: vec![], account: Account::Unknown }
    }
}

pub struct Run {
    pub node: u64,
    /// The pipeline as it was when the run was asked for: a quote is for
    /// THIS, and an edit since makes it stale.
    pub pipeline: Pipeline,
    pub tick: i64,
    pub phase: Phase,
    pub steps: BTreeMap<usize, StepState>,
}

pub enum Phase {
    /// Rendering the node's inputs; `todo` = ports still to render.
    Capturing { inputs: Inputs, todo: Vec<usize> },
    Quoting { plan: Plan, inputs: Inputs, rx: Receiver<Result<Quote, String>> },
    /// Waiting for "pay & run".
    Ready { plan: Plan, inputs: Inputs, quote: Result<Quote, String> },
    Running { job: run::Job },
    Failed(String),
    Done(String),
}

impl Run {
    pub(crate) fn active(&self) -> bool {
        !matches!(self.phase, Phase::Failed(_) | Phase::Done(_))
    }
}

/// What the panel needs to know, snapshotted before the node borrow.
pub struct View {
    pub route: Route,
    /// How a billed step would be paid, in words; `Err` = it cannot be.
    pub bill: Result<String, String>,
    pub fal: Option<Origin>,
    pub relay: Option<Origin>,
    pub comfy_url: String,
    pub state: ViewState,
    pub steps: BTreeMap<usize, StepState>,
    /// Another node's run is going.
    pub busy_elsewhere: bool,
    /// What the node's inputs have wired, for the hints.
    pub has_input: bool,
    pub has_matte: bool,
    /// The text on the node's prompt port (a wired String node), if any.
    pub prompt: Option<String>,
    /// The picked take's file is not on this machine.
    pub take_missing: bool,
    /// The comp's frame rate, for a clip's frame count.
    pub fps: f32,
}

pub enum ViewState {
    Idle,
    Capturing,
    Quoting,
    Ready { quote: Result<Quote, String>, billed: usize, stale: bool },
    Running,
    Failed(String),
    Done(String),
}

/// What the panel asks for; applied after the node borrow.
pub enum Action {
    Prepare,
    Confirm,
    Cancel,
    Dismiss,
    /// The picked take changed (picked, forgotten): reload the reader.
    Show,
    SetRoute(Route),
    SaveKey(Key, String),
    ForgetKey(Key),
    SaveComfy(String),
    Refresh,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Fal,
    Relay,
}

// ------------------------------------------------------------ panel

fn small(t: impl Into<String>) -> egui::RichText {
    egui::RichText::new(t.into()).small()
}

fn state_words(s: &StepState) -> String {
    match s {
        StepState::Cached => "cached — not billed".into(),
        StepState::Sending => "sending".into(),
        StepState::Queued(Some(n)) => format!("queued, {n} ahead"),
        StepState::Queued(None) => "queued".into(),
        StepState::Running => "running".into(),
        StepState::Fetching => "fetching the result".into(),
        StepState::Done => "done".into(),
    }
}

/// The node's properties: steps on the left, billing + run + takes right.
pub fn panel(ui: &mut Ui, node: u64, p: &mut GenAiParams, v: &View) -> Option<Action> {
    let mut act = None;
    ui.columns(2, |cols| {
        if p.model.is_some() {
            model_column(&mut cols[0], node, p, v);
        } else {
            pipeline_column(&mut cols[0], p, v);
        }
        if let Some(a) = run_column(&mut cols[1], node, p, v) {
            act = Some(a);
        }
    });
    act
}

/// A one-model node's left column: its inputs, then the model's own
/// fields. No step list, no model menu: the node IS the model.
fn model_column(ui: &mut Ui, node: u64, p: &mut GenAiParams, v: &View) {
    let pl = &mut p.pipeline;
    let Some(g) = pl.model_step() else {
        ui.colored_label(theme::error(), small("this node's pipeline has lost its model step"));
        return;
    };
    for i in 0..pl.steps.len() {
        if i == g || matches!(pl.steps[i].op, Op::Output) {
            continue;
        }
        let id = move |what: &str| egui::Id::new(("genai", node, i, what.to_string()));
        if matches!(pl.steps[i].op, Op::Prompt { .. }) && v.prompt.is_some() {
            // A wired String node wins over the typed prompt (the run reads
            // the port instead), so say so rather than offer a dead box.
            op_fields(ui, &id, &mut Op::PromptIn, v, false);
        } else {
            op_fields(ui, &id, &mut pl.steps[i].op, v, false);
        }
        ui.add_space(2.0);
    }
    let id = move |what: &str| egui::Id::new(("genai", node, g, what.to_string()));
    op_fields(ui, &id, &mut pl.steps[g].op, v, false);
}

/// A GenAI Pipeline's left column: what its steps are, in run order, and
/// how to open them. The steps themselves are edited inside (Enter).
fn pipeline_column(ui: &mut Ui, p: &GenAiParams, v: &View) {
    let pl = &p.pipeline;
    ui.label(small("steps (Enter opens them, like a Noise)"));
    match pl.output().map(|o| pl.plan(o)) {
        Some(Ok(order)) => {
            for i in order {
                let s = &pl.steps[i];
                let state = v.steps.get(&i).map(state_words).unwrap_or_default();
                ui.horizontal(|ui| {
                    ui.label(small(s.op.name()));
                    if let Op::Prompt { text } = &s.op {
                        ui.weak(small(text.chars().take(40).collect::<String>()));
                    }
                    if !state.is_empty() {
                        ui.weak(small(state));
                    }
                });
            }
        }
        Some(Err(e)) => {
            ui.colored_label(theme::warn(), small(e));
        }
        None => {
            ui.colored_label(theme::warn(), small("no Output step: open the pipeline and add one"));
        }
    }
}

/// One step's fields. `pick_model` offers the model menu on a model step
/// (the free-form pipeline); a one-model node passes false. Returns the
/// model picked from that menu, if one was.
fn op_fields(ui: &mut Ui, id: &dyn Fn(&str) -> egui::Id, op: &mut Op, v: &View, pick_model: bool) -> Option<&'static str> {
    let mut model_change: Option<&'static str> = None;
    match op {
        Op::Prompt { text } => {
            ui.add(
                egui::TextEdit::multiline(text)
                    .id(id("prompt"))
                    .hint_text("describe the picture, or the change")
                    .desired_rows(6)
                    .desired_width(f32::INFINITY),
            );
        }
        Op::PromptIn => {
            ui.label(small("the text on the node's prompt port: wire a String node into it"));
            match &v.prompt {
                Some(t) if !t.is_empty() => {
                    ui.weak(small(t.chars().take(200).collect::<String>()));
                }
                Some(_) => {
                    ui.colored_label(theme::warn(), small("the wired String node is empty"));
                }
                None => {
                    ui.colored_label(theme::warn(), small("no String node is wired into the prompt port"));
                }
            }
        }
        Op::Reference => {
            ui.label(small("the node's image input, rendered at the playhead when you run"));
            ui.weak(small(format!(
                "sent as sRGB at up to {REF_MP} MP: values above 1.0 come back clipped"
            )));
            if !v.has_input {
                ui.colored_label(theme::warn(), small("nothing is wired into the image input"));
            }
        }
        Op::Matte => {
            ui.label(small("the node's matte input at the playhead: white = where to work"));
            if !v.has_matte {
                ui.colored_label(theme::warn(), small("nothing is wired into the matte input"));
            }
        }
        Op::ClipIn { seconds } => {
            ui.label(small("the node's image input over a range from the playhead, sent as an mp4"));
            ui.horizontal(|ui| {
                ui.label(small("length"));
                ui.add(egui::DragValue::new(seconds).range(0.5..=30.0).speed(0.05).suffix(" s"));
            });
            ui.weak(small(format!(
                "{:.0} frames at the comp's rate, sRGB, up to {CLIP_SHORT} px on the short side",
                (*seconds * v.fps).round()
            )));
            if !v.has_input {
                ui.colored_label(theme::warn(), small("nothing is wired into the image input"));
            }
            if !clip_ok() {
                ui.colored_label(theme::error(), small(NO_CLIP));
            }
        }
        Op::MatteClip => {
            ui.label(small("the node's matte input over the Input Clip's range: white = where to work"));
            ui.weak(small("wire a Roto, Paint or Sammie into the node's matte input"));
            if !v.has_matte {
                ui.colored_label(theme::warn(), small("nothing is wired into the matte input"));
            }
            if !clip_ok() {
                ui.colored_label(theme::error(), small(NO_CLIP));
            }
        }
        Op::Output => {
            ui.label(small("the node's picture: wire the step whose result you want"));
            ui.weak(small("stretched over the node's input format, so an edit lines up with its plate"));
        }
        Op::Generate { model, seed, opts } => {
            let cur = catalog::get(model);
            if !pick_model {
                ui.label(egui::RichText::new(cur.map(|m| m.label).unwrap_or("unknown model")).strong());
            } else {
                egui::ComboBox::from_id_salt(id("model"))
                    .width(ui.available_width())
                    .selected_text(small(cur.map(|m| m.label).unwrap_or("unknown model")))
                    .show_ui(ui, |ui| {
                        for t in Task::ALL {
                            ui.label(small(t.label()).weak());
                            for m in catalog::for_task(t).filter(|m| offered(m)) {
                                if ui.add(egui::Button::selectable(cur.is_some_and(|c| c.id == m.id), m.label)).clicked() {
                                    model_change = Some(m.id);
                                }
                            }
                        }
                    });
            }
            match cur {
                Some(m) => {
                    ui.weak(small(m.about));
                    let lic = small(format!("licence: {}", m.licence));
                    if m.commercial {
                        ui.weak(lic);
                    } else {
                        ui.colored_label(theme::cost_nc(), lic);
                        ui.colored_label(theme::cost_nc(), small("non-commercial: tagged downstream, and a Render warns"));
                    }
                    ui.horizontal(|ui| {
                        ui.add_sized([crate::widgets::LABEL_W, 16.0], egui::Label::new(small("seed")));
                        ui.add(egui::DragValue::new(seed).update_while_editing(false).speed(1.0));
                        if ui.small_button("new seed").on_hover_text("a variation: a new request, billed again").clicked() {
                            *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345) % 1_000_000;
                        }
                    });
                    if !m.dialect.sends_seed() {
                        ui.weak(small("this model takes no seed: a new one asks for a variation"));
                    }
                    match m.task {
                        Task::TextToImage => {
                            let (lo, hi) = m.dialect.mp_range();
                            let (lo, hi) = (lo.max(0.25), hi.min(2.0).max(lo.max(0.25)));
                            opts.megapixels = opts.megapixels.clamp(lo, hi);
                            crate::widgets::plain_slider(ui, "megapixels", &mut opts.megapixels, lo, hi, false);
                            ui.weak(small("at the comp's aspect"));
                        }
                        Task::ImageToVideo => {
                            if let catalog::Dialect::Seedance { max_seconds, top_res } = m.dialect {
                                let mut secs = opts.seconds.clamp(4, max_seconds) as f32;
                                crate::widgets::plain_slider(ui, "seconds", &mut secs, 4.0, max_seconds as f32, false);
                                opts.seconds = secs.round() as u32;
                                ui.horizontal(|ui| {
                                    ui.add_sized([crate::widgets::LABEL_W, 16.0], egui::Label::new(small("size")));
                                    for r in [480u32, 720, 1080].into_iter().filter(|&r| r <= top_res) {
                                        if ui.add(egui::Button::selectable(opts.res == r, small(format!("{r}p")))).clicked() {
                                            opts.res = r;
                                        }
                                    }
                                });
                                ui.weak(small("billed per second; no audio; plays from the frame it was made at"));
                            }
                            if !video_ok() {
                                ui.colored_label(theme::error(), small(NO_VIDEO));
                            }
                        }
                        Task::Upscale => {
                            crate::widgets::plain_slider(ui, "scale", &mut opts.scale, 1.0, 4.0, false);
                        }
                        _ => {}
                    }
                    ui.horizontal(|ui| {
                        ui.add_sized([crate::widgets::LABEL_W, 16.0], egui::Label::new(small("extra")))
                            .on_hover_text("a JSON object merged over the request, for fields this build does not model");
                        ui.add(
                            egui::TextEdit::singleline(&mut opts.extra)
                                .id(id("extra"))
                                .hint_text("{\"guidance_scale\": 3.5}")
                                .desired_width(f32::INFINITY),
                        );
                    });
                    if !opts.extra.trim().is_empty()
                        && serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&opts.extra).is_err()
                    {
                        ui.colored_label(theme::error(), small("extra is not a JSON object"));
                    }
                }
                None => {
                    ui.colored_label(theme::error(), small(format!("this build does not know \"{model}\"")));
                }
            }
        }
        Op::Comfy { workflow, seed, commercial } => {
            ui.label(small(format!("a local ComfyUI at {}", v.comfy_url)));
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(workflow)
                        .id(id("wf"))
                        .hint_text("workflow_api.json")
                        .desired_width(ui.available_width() - 60.0),
                );
                if ui.small_button("browse…").clicked() {
                    if let Some(f) = crate::filedialog::FileDialog::new().add_filter("ComfyUI API workflow", &["json"]).pick_file() {
                        *workflow = f.to_string_lossy().into_owned();
                    }
                }
            });
            ui.weak(small("export with \"Save (API format)\"; mark inputs {{prompt}} {{seed}} {{image}} {{mask}} {{width}} {{height}}"));
            ui.horizontal(|ui| {
                ui.add_sized([crate::widgets::LABEL_W, 16.0], egui::Label::new(small("seed")));
                ui.add(egui::DragValue::new(seed).update_while_editing(false).speed(1.0));
            });
            ui.checkbox(commercial, small("its models are free for commercial use (my word)"));
        }
    }
    model_change
}

fn run_column(ui: &mut Ui, node: u64, p: &mut GenAiParams, v: &View) -> Option<Action> {
    let mut act = None;
    // --- who pays
    // Wobble credits are the `commerce` feature; without it the one route
    // is the user's own key, and there is nothing to pick.
    if cfg!(feature = "commerce") {
    ui.horizontal(|ui| {
        ui.label(small("bill"));
        #[cfg(feature = "commerce")]
        if ui.add(egui::Button::selectable(v.route == Route::Relay, small("wobble credits"))).clicked() && v.route != Route::Relay {
            act = Some(Action::SetRoute(Route::Relay));
        }
        if ui.add(egui::Button::selectable(v.route == Route::Own, small("my fal key"))).clicked() && v.route != Route::Own {
            act = Some(Action::SetRoute(Route::Own));
        }
    });
    }
    match &v.bill {
        Ok(s) => {
            ui.horizontal(|ui| {
                ui.weak(small(s));
                if v.route == Route::Relay && ui.small_button("refresh").clicked() {
                    act = Some(Action::Refresh);
                }
            });
        }
        Err(e) => {
            ui.colored_label(theme::warn(), small(e));
        }
    }
    // A GLOBAL id (not scoped to the column), so the fit test can open it.
    let kid = egui::Id::new(("genai-keys", node));
    let keys = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), kid, false);
    keys.show_header(ui, |ui| {
        ui.label(small("keys…"));
    })
    .body(|ui| {
        if let Some(a) = key_row(ui, "fal key", Key::Fal, v.fal) {
            act = Some(a);
        }
        if cfg!(feature = "commerce") {
            if let Some(a) = key_row(ui, "relay token", Key::Relay, v.relay) {
                act = Some(a);
            }
        }
        let cid = egui::Id::new("genai-comfy-url");
        let mut url = ui.data_mut(|d| d.get_temp::<String>(cid)).unwrap_or_else(|| v.comfy_url.clone());
        ui.horizontal(|ui| {
            ui.add_sized([60.0, 16.0], egui::Label::new(small("ComfyUI")));
            // Button first from the right, so the field takes exactly
            // what is left and the row never runs past the tile.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("save").clicked() {
                    act = Some(Action::SaveComfy(url.clone()));
                }
                ui.add(egui::TextEdit::singleline(&mut url).desired_width(ui.available_width()));
            });
        });
        ui.data_mut(|d| d.insert_temp(cid, url));
        ui.weak(small(format!("kept in {}, readable only by you — never in a setup", config::config_path().display())));
    });
    ui.separator();

    // --- the run
    match &v.state {
        ViewState::Idle | ViewState::Failed(_) | ViewState::Done(_) => {
            if let ViewState::Failed(e) = &v.state {
                ui.colored_label(theme::error(), small(e));
            }
            if let ViewState::Done(s) = &v.state {
                ui.weak(small(s));
            }
            let busy = v.busy_elsewhere;
            let b = ui.add_enabled(!busy, egui::Button::new("run…"));
            let b = if busy {
                b.on_disabled_hover_text("another GenAI node is running")
            } else {
                b.on_hover_text("renders the inputs at the playhead and asks the price; nothing is billed until you confirm")
            };
            if b.clicked() {
                act = Some(Action::Prepare);
            }
        }
        ViewState::Capturing => {
            ui.label(small("rendering the inputs at the playhead…"));
            if ui.small_button("cancel").clicked() {
                act = Some(Action::Cancel);
            }
        }
        ViewState::Quoting => {
            ui.label(small("asking the price…"));
            if ui.small_button("cancel").clicked() {
                act = Some(Action::Cancel);
            }
        }
        ViewState::Ready { quote, billed, stale } => {
            let calls = if *billed == 1 { "1 billed call".to_string() } else { format!("{billed} billed calls") };
            match quote {
                Ok(q) => {
                    let price = match q.total() {
                        Some(t) if q.exact() => service::usd(t),
                        Some(t) => format!("≈ {}", service::usd(t)),
                        None => "price unknown".into(),
                    };
                    ui.label(small(format!("{calls}: {price}")));
                    let unpriced = q.items.iter().filter(|p| p.is_none()).count();
                    if unpriced > 0 {
                        ui.colored_label(
                            theme::warn(),
                            small(format!("{unpriced} call(s) have no price from the service: running bills what it bills")),
                        );
                    }
                    if let Some(b) = q.balance_usd.filter(|_| cfg!(feature = "commerce")) {
                        ui.weak(small(format!("balance {}", service::usd(b))));
                    }
                }
                Err(e) => {
                    ui.label(small(calls));
                    ui.colored_label(theme::warn(), small(format!("no price: {e}")));
                }
            }
            if *stale {
                ui.colored_label(theme::warn(), small("the pipeline changed since: run… again"));
            }
            ui.horizontal(|ui| {
                let label = if quote.is_ok() { "pay & run" } else { "run without a price" };
                if ui.add_enabled(!*stale, egui::Button::new(label)).clicked() {
                    act = Some(Action::Confirm);
                }
                if ui.small_button("cancel").clicked() {
                    act = Some(Action::Cancel);
                }
            });
        }
        ViewState::Running => {
            for (i, s) in &v.steps {
                let name = p.pipeline.steps.get(*i).map(|s| s.op.label()).unwrap_or_default();
                ui.label(small(format!("{name}: {}", state_words(s))));
            }
            if ui.small_button("cancel").on_hover_text("stops waiting and cancels the remote job").clicked() {
                act = Some(Action::Cancel);
            }
        }
    }
    if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("ok").clicked() {
        act = Some(Action::Dismiss);
    }
    ui.separator();

    // --- takes
    let pl = &mut p.pipeline;
    if pl.takes.is_empty() {
        ui.weak(small("no takes yet: the node passes its input through"));
    } else {
        ui.label(small("takes"));
        let mut forget: Option<usize> = None;
        egui::ScrollArea::vertical().id_salt(("genai-takes", node)).max_height(96.0).show(ui, |ui| {
            for k in (0..pl.takes.len()).rev() {
                let t = &pl.takes[k];
                let mut text = format!("{} · {}", k + 1, if t.label.is_empty() { "take" } else { &t.label });
                if let Some(u) = t.usd {
                    text.push_str(&format!(" · {}", service::usd(u)));
                }
                ui.horizontal(|ui| {
                    let mut rt = small(text);
                    if !t.commercial {
                        rt = rt.color(theme::cost_nc());
                    }
                    if ui.add(egui::Button::selectable(k == pl.take, rt).truncate()).clicked() && k != pl.take {
                        pl.take = k;
                        act = Some(Action::Show);
                    }
                    if ui.small_button("×").on_hover_text("forget this take (its file stays on disk)").clicked() {
                        forget = Some(k);
                    }
                });
            }
        });
        if let Some(k) = forget {
            pl.takes.remove(k);
            if pl.take > k || pl.take >= pl.takes.len() {
                pl.take = pl.take.saturating_sub(1);
            }
            act = Some(Action::Show);
        }
        if v.take_missing {
            ui.colored_label(theme::warn(), small("this take's file is not on this machine; running again bills again"));
        }
        if pl.picked().is_some_and(|t| t.is_video()) && !video_ok() {
            // Loud, never a silent black: the node passes its input through.
            ui.colored_label(theme::error(), small(NO_VIDEO));
        }
    }
    if !p.commercial() {
        ui.colored_label(theme::cost_nc(), small("non-commercial pixels: tagged downstream"));
    }
    act
}

/// A secret: masked when set, a password field when not.
fn key_row(ui: &mut Ui, label: &str, which: Key, origin: Option<Origin>) -> Option<Action> {
    let mut act = None;
    let id = egui::Id::new(("genai-key-edit", which == Key::Fal));
    ui.horizontal(|ui| {
        ui.add_sized([60.0, 16.0], egui::Label::new(small(label)));
        match origin {
            Some(Origin::Env(var)) => {
                ui.weak(small(format!("from {var}")));
            }
            Some(_) => {
                ui.weak(small("saved"));
                if ui.small_button("forget").clicked() {
                    act = Some(Action::ForgetKey(which));
                }
            }
            None => {
                let mut s = ui.data_mut(|d| d.get_temp::<String>(id)).unwrap_or_default();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("save").clicked() && !s.trim().is_empty() {
                        act = Some(Action::SaveKey(which, s.trim().to_string()));
                        s.clear();
                    }
                    ui.add(egui::TextEdit::singleline(&mut s).password(true).desired_width(ui.available_width()));
                });
                ui.data_mut(|d| d.insert_temp(id, s));
            }
        }
    });
    act
}

// ------------------------------------------------------------ the app side

/// Working-space RGBA -> linear Rec.709, the space `genai::image` encodes
/// from.
/// The text on `node`'s prompt port: a String node wired in, through Dots.
fn wired_prompt(g: &crate::editor::EditorGraph, node: u64) -> Option<String> {
    let mut src = g.input_of(node, 2)?;
    for _ in 0..64 {
        match &g.node(src)?.kind {
            NodeKind::Prompt(p) => return Some(p.text.clone()),
            NodeKind::Dot { .. } => src = g.input_of(src, 0)?,
            _ => return None,
        }
    }
    None
}

/// A clip capture's frames, encoded to an mp4 for the run.
fn encode_clip(frames: Vec<(Vec<u8>, u32, u32)>, fps: f32) -> Result<Clip, String> {
    let Some((w, h)) = frames.first().map(|f| (f.1, f.2)) else {
        return Err("the input did not render".into());
    };
    if frames.iter().any(|f| (f.1, f.2) != (w, h)) {
        return Err("the input changed size mid-clip".into());
    }
    let n = frames.len() as u32;
    #[cfg(feature = "movie")]
    {
        let path = std::env::temp_dir().join(format!("wobble-genai-clip-{}.mp4", std::process::id()));
        let px: Vec<Vec<u8>> = frames.into_iter().map(|f| f.0).collect();
        let r = engine::movie::encode_mp4(&path, &px, w, h, fps)
            .map_err(|e| format!("encoding the clip: {e:#}"))
            .and_then(|_| std::fs::read(&path).map_err(|e| format!("reading the clip back: {e}")));
        let _ = std::fs::remove_file(&path);
        r.map(|bytes| Clip { bytes, w, h, frames: n, fps })
    }
    #[cfg(not(feature = "movie"))]
    {
        let _ = (n, fps);
        Err(NO_CLIP.into())
    }
}

fn to_rec709(rgba: &mut [f32], working: engine::color::Colorspace) {
    let t = engine::color::Transform::new(working, engine::color::Colorspace::LINEAR_REC709);
    if t.is_identity() {
        return;
    }
    for px in rgba.chunks_exact_mut(4) {
        let c = t.apply([px[0], px[1], px[2]]);
        px[..3].copy_from_slice(&c);
    }
}

impl Gfx {
    pub(crate) fn genai_view(&self, node: u64) -> View {
        let cfg = &self.jobs.genai.cfg;
        let bill = match cfg.route {
            #[cfg(not(feature = "commerce"))]
            Route::Relay => Err("not available in this build".to_string()),
            #[cfg(feature = "commerce")]
            Route::Relay => match (&cfg.relay_token, &self.jobs.genai.account) {
                (None, _) => Err("no wobble account on this machine: add a relay token under keys…".to_string()),
                (Some(_), Account::Known(Ok((who, bal)))) => {
                    Ok(format!("wobble credits · {who} · {}", service::usd(*bal)))
                }
                (Some(_), Account::Known(Err(e))) => Err(format!("relay: {e}")),
                (Some(_), _) => Ok("wobble credits".to_string()),
            },
            Route::Own => match &cfg.fal_key {
                Some((k, _)) => Ok(format!("fal, your key {}", config::masked(k))),
                None => Err("no fal key on this machine: set FAL_KEY or add one under keys…".to_string()),
            },
        };
        let mine = self.jobs.genai.run.as_ref().filter(|r| r.node == node);
        let state = match mine.map(|r| &r.phase) {
            None => ViewState::Idle,
            Some(Phase::Capturing { .. }) => ViewState::Capturing,
            Some(Phase::Quoting { .. }) => ViewState::Quoting,
            Some(Phase::Ready { plan, quote, inputs }) => {
                let r = mine.unwrap();
                let stale = self.genai_runnable(node).is_none_or(|p| p.steps != r.pipeline.steps)
                    || self.genai_prompt_moved(node, &r.pipeline, inputs);
                ViewState::Ready { quote: quote.clone(), billed: plan.billed(&r.pipeline).len(), stale }
            }
            Some(Phase::Running { .. }) => ViewState::Running,
            Some(Phase::Failed(e)) => ViewState::Failed(e.clone()),
            Some(Phase::Done(s)) => ViewState::Done(s.clone()),
        };
        let take_missing = self
            .egraph
            .node(node)
            .and_then(|n| params(&n.kind))
            .and_then(|p| p.pipeline.picked())
            .is_some_and(|t| !t.path(&config::results_dir()).exists());
        View {
            route: cfg.route,
            bill,
            fal: cfg.fal_key.as_ref().map(|k| k.1),
            relay: cfg.relay_token.as_ref().map(|k| k.1),
            comfy_url: cfg.comfy_url.clone(),
            state,
            steps: mine.map(|r| r.steps.clone()).unwrap_or_default(),
            busy_elsewhere: self.jobs.genai.run.as_ref().is_some_and(|r| r.node != node && r.active()),
            has_input: self.egraph.input_of(node, 0).is_some(),
            has_matte: self.egraph.input_of(node, 1).is_some(),
            prompt: self.genai_wired_prompt(node),
            take_missing,
            fps: self.fps,
        }
    }

    pub(crate) fn genai_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Prepare => self.genai_prepare(node),
            Action::Confirm => self.genai_confirm(node),
            Action::Cancel => {
                if let Some(r) = self.jobs.genai.run.take() {
                    if let Phase::Running { job } = &r.phase {
                        job.cancel();
                    }
                    if matches!(r.phase, Phase::Capturing { .. }) {
                        // The capture is a one-frame bake: drop it.
                        self.graph_bake = None;
                        self.jobs.genai.captured.clear();
                    }
                    // A running job's thread sees the cancel and cancels
                    // the remote request; nobody is listening for its end.
                }
            }
            Action::Dismiss => {
                if self.jobs.genai.run.as_ref().is_some_and(|r| !r.active()) {
                    self.jobs.genai.run = None;
                }
            }
            Action::Show => self.genai_show(node),
            Action::SetRoute(route) => {
                let _ = config::update(|f| f.route = route);
                self.genai_reload_config();
            }
            Action::SaveKey(k, v) => {
                let r = config::update(|f| match k {
                    Key::Fal => f.fal_key = Some(v),
                    Key::Relay => f.relay_token = Some(v),
                });
                if let Err(e) = r {
                    self.render_status = Some(format!("could not save the key: {e:#}"));
                }
                self.genai_reload_config();
            }
            Action::ForgetKey(k) => {
                let _ = config::update(|f| match k {
                    Key::Fal => f.fal_key = None,
                    Key::Relay => f.relay_token = None,
                });
                self.genai_reload_config();
            }
            Action::SaveComfy(url) => {
                let url = url.trim().to_string();
                let _ = config::update(|f| f.comfy_url = (!url.is_empty()).then_some(url));
                self.genai_reload_config();
            }
            Action::Refresh => self.jobs.genai.account = Account::Unknown,
        }
    }

    fn genai_reload_config(&mut self) {
        self.jobs.genai.cfg = Config::load();
        self.jobs.genai.account = Account::Unknown;
    }

    fn genai_fail(&mut self, node: u64, pipeline: Pipeline, msg: impl Into<String>) {
        self.jobs.genai.run = Some(Run {
            node,
            pipeline,
            tick: self.timeline_pos as i64,
            phase: Phase::Failed(msg.into()),
            steps: BTreeMap::new(),
        });
    }

    fn genai_prepare(&mut self, node: u64) {
        if self.jobs.genai.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        let Some(pl) = self.genai_runnable(node) else { return };
        // Everything that can be said without rendering or sending, first.
        let Some(out) = pl.output() else {
            return self.genai_fail(node, pl, "the pipeline has no Output step");
        };
        if let Err(e) = pl.plan(out) {
            return self.genai_fail(node, pl, e);
        }
        let video_step = pl.plan(out).unwrap().into_iter().any(|i| {
            matches!(&pl.steps[i].op, Op::Generate { model, .. }
                if catalog::get(model).is_some_and(|m| m.task.is_video()))
        });
        if video_step && !video_ok() {
            return self.genai_fail(node, pl, NO_VIDEO);
        }
        if pl.plan(out).unwrap().iter().any(|&i| pl.steps[i].op.is_remote()) {
            if let Err(e) = self.jobs.genai.cfg.can_bill() {
                return self.genai_fail(node, pl, e);
            }
        }
        let prompt = self.genai_wired_prompt(node);
        if pl.wants_prompt() && prompt.is_none() {
            return self.genai_fail(node, pl, "the pipeline reads the node's prompt: wire a String node into it");
        }
        let prompt = prompt.filter(|_| pl.wants_prompt());
        if (pl.wants_clip() || pl.wants_matte_clip()) && !clip_ok() {
            return self.genai_fail(node, pl, NO_CLIP);
        }
        let mut todo = vec![];
        for (port, wants, what) in [
            (0, pl.wants_reference(), "image"),
            (1, pl.wants_matte(), "matte"),
            (CLIP, pl.wants_clip(), "image"),
            (MATTE_CLIP, pl.wants_matte_clip(), "matte"),
        ] {
            if wants {
                if self.egraph.input_of(node, input_of_port(port)).is_none() {
                    return self.genai_fail(node, pl, format!("the pipeline reads the node's {what}: wire one in"));
                }
                todo.push(port);
            }
        }
        if !todo.is_empty() {
            if let Some(up) = self.genai_unrun_upstream(node) {
                return self.genai_fail(node, pl, format!("{up} upstream has no take yet: run it first"));
            }
        }
        if !todo.is_empty() && self.graph_bake.is_some() {
            return self.genai_fail(node, pl, "a bake is running: wait for it, then run");
        }
        let Some(comp) = self.comp.output_size() else {
            return self.genai_fail(node, pl, "nothing is rendering yet");
        };
        let tick = self.timeline_pos as i64;
        self.jobs.genai.captured.clear();
        self.jobs.genai.clip_frames.clear();
        self.jobs.genai.run = Some(Run {
            node,
            pipeline: pl,
            tick,
            phase: Phase::Capturing { inputs: Inputs { prompt, comp, tick, ..Default::default() }, todo },
            steps: BTreeMap::new(),
        });
        self.genai_capture_next();
    }

    /// Render the next input the run needs, or move on to the plan.
    fn genai_capture_next(&mut self) {
        let Some(r) = &self.jobs.genai.run else { return };
        let Phase::Capturing { todo, .. } = &r.phase else { return };
        let (node, tick, next, pl) = (r.node, r.tick, todo.first().copied(), r.pipeline.clone());
        match next {
            Some(port) => {
                let Some(input) = self.egraph.input_of(node, input_of_port(port)) else {
                    return self.genai_fail(node, pl, "an input was unwired mid-run");
                };
                let frames = if port >= CLIP { self.genai_clip_frames(&pl) } else { 1 };
                if let Err(e) = self.start_capture(node, input, port, tick, frames) {
                    self.genai_fail(node, pl, e);
                }
            }
            None => self.genai_plan(),
        }
    }

    /// How many frames a run's clips are: its length at the comp's rate.
    fn genai_clip_frames(&self, pl: &Pipeline) -> u64 {
        ((pl.clip_seconds() * self.fps.max(1.0)).round() as u64).max(1)
    }

    /// A bake of `input` over `frames` ticks from `tick`, whose pixels land
    /// in `genai.captured` (one frame) or `genai.clip_frames` (a clip)
    /// instead of a file (`BakeSink::Capture`).
    fn start_capture(&mut self, node: u64, input: u64, port: usize, tick: i64, frames: u64) -> Result<(), String> {
        let key = self.subtree_key_of(input).ok_or("the input is not renderable yet")?;
        let (w, h) = self.comp.output_size().ok_or("nothing is rendering yet")?;
        let tick = tick.max(0) as u64;
        let pb = self.playback.as_mut().ok_or("load a sequence first")?;
        self.playing = false;
        pb.player.seek(tick);
        let info = engine::decode::ImageInfo { width: w, height: h };
        let readback = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("genai-capture"),
            size: info.staging_bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.load_errors.remove(&node);
        self.graph_bake = Some(crate::SubtreeBake {
            node,
            input,
            key,
            dir: std::env::temp_dir(),
            from: tick,
            len: tick + frames.max(1),
            next: tick,
            readback,
            info,
            sink: crate::BakeSink::Capture { port },
        });
        Ok(())
    }

    /// The bake loop finished a capture: encode what it left and go on.
    pub(crate) fn genai_capture_finished(&mut self, node: u64) {
        let working = self.egraph.working;
        let got = std::mem::take(&mut self.jobs.genai.captured);
        let Some(r) = self.jobs.genai.run.as_mut().filter(|r| r.node == node) else { return };
        let Phase::Capturing { inputs, todo } = &mut r.phase else { return };
        let Some(port) = todo.first().copied() else { return };
        if port >= CLIP {
            let frames: Vec<(Vec<u8>, u32, u32)> = std::mem::take(&mut self.jobs.genai.clip_frames)
                .into_iter()
                .filter(|c| c.0 == node && c.1 == port)
                .map(|c| (c.2, c.3, c.4))
                .collect();
            match encode_clip(frames, self.fps) {
                Ok(clip) => {
                    if port == CLIP {
                        inputs.clip = Some(clip);
                    } else {
                        inputs.matte_clip = Some(clip);
                    }
                    todo.remove(0);
                    self.genai_capture_next();
                }
                Err(e) => r.phase = Phase::Failed(e),
            }
            return;
        }
        let Some((_, _, mut px, w, h)) = got.into_iter().find(|c| c.0 == node && c.1 == port) else {
            r.phase = Phase::Failed("the input did not render".into());
            return;
        };
        let png = if port == 0 {
            to_rec709(&mut px, working);
            let (small, sw, sh) = genai::image::shrink_rgba(&px, w, h, REF_MP);
            genai::image::png_srgb_rgb(&small, sw, sh).map(|b| Png { bytes: b, w: sw, h: sh })
        } else {
            // Coverage rides every channel of a matte; take red.
            let cov: Vec<f32> = px.chunks_exact(4).map(|p| p[0]).collect();
            let (small, sw, sh) = genai::image::shrink_mono(&cov, w, h, REF_MP);
            genai::image::png_matte(&small, sw, sh).map(|b| Png { bytes: b, w: sw, h: sh })
        };
        match png {
            Ok(png) => {
                if port == 0 {
                    inputs.reference = Some(png);
                } else {
                    inputs.matte = Some(png);
                }
                todo.remove(0);
                self.genai_capture_next();
            }
            Err(e) => r.phase = Phase::Failed(format!("encoding the input: {e:#}")),
        }
    }

    /// Key the steps, find the cache, and either ask the price or — with
    /// nothing billed — run straight away.
    fn genai_plan(&mut self) {
        let Some(r) = self.jobs.genai.run.as_mut() else { return };
        let Phase::Capturing { inputs, .. } = std::mem::replace(&mut r.phase, Phase::Failed(String::new())) else {
            return;
        };
        let dir = config::results_dir();
        let plan = match run::plan(&r.pipeline, &inputs, &dir) {
            Ok(p) => p,
            Err(e) => {
                r.phase = Phase::Failed(e);
                return;
            }
        };
        if plan.billed(&r.pipeline).is_empty() {
            let job = run::start(
                r.pipeline.clone(),
                plan,
                inputs,
                run::Services::from_config(&self.jobs.genai.cfg),
                dir,
                vec![],
            );
            r.phase = Phase::Running { job };
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (pl, pl2, ins) = (r.pipeline.clone(), plan.clone(), inputs.clone());
        let cfg = self.jobs.genai.cfg.clone();
        let spawned = std::thread::Builder::new().name("genai-quote".into()).spawn(move || {
            let q = run::service_for(&cfg).and_then(|svc| {
                run::quote(&pl, &pl2, &ins, svc.as_ref(), &genai::http::Http::new()).map_err(|e| format!("{e:#}"))
            });
            let _ = tx.send(q);
        });
        r.phase = match spawned {
            Ok(_) => Phase::Quoting { plan, inputs, rx },
            Err(_) => Phase::Failed("running a pipeline needs the desktop app".into()),
        };
    }

    fn genai_confirm(&mut self, node: u64) {
        let current = self.genai_runnable(node).map(|p| p.steps);
        let moved = match self.jobs.genai.run.as_ref().filter(|r| r.node == node) {
            Some(Run { pipeline, phase: Phase::Ready { inputs, .. }, .. }) => {
                self.genai_prompt_moved(node, pipeline, inputs)
            }
            _ => false,
        };
        let Some(r) = self.jobs.genai.run.as_mut().filter(|r| r.node == node) else { return };
        if moved || current.as_ref() != Some(&r.pipeline.steps) {
            return; // stale: the panel says so and disables the button
        }
        let Phase::Ready { plan, inputs, quote } = std::mem::replace(&mut r.phase, Phase::Failed(String::new())) else {
            return;
        };
        // What was shown is what may be charged, per billed step.
        let approved: run::Approved = match &quote {
            Ok(q) => plan
                .billed(&r.pipeline)
                .into_iter()
                .zip(q.items.iter())
                .filter_map(|(i, p)| p.as_ref().map(|p| (i, p.usd + 1e-6)))
                .collect(),
            Err(_) => vec![],
        };
        let job = run::start(
            r.pipeline.clone(),
            plan,
            inputs,
            run::Services::from_config(&self.jobs.genai.cfg),
            config::results_dir(),
            approved,
        );
        r.phase = Phase::Running { job };
    }

    /// Once a redraw: quotes landing, run progress, the account balance.
    pub(crate) fn genai_poll(&mut self) {
        // The relay balance, fetched once per config change.
        #[cfg(feature = "commerce")]
        if self.jobs.genai.cfg.route == Route::Relay && self.jobs.genai.cfg.relay_token.is_some() {
            match &self.jobs.genai.account {
                Account::Unknown => {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let (base, tok) = (self.jobs.genai.cfg.relay_url.clone(), self.jobs.genai.cfg.relay_token.clone().unwrap().0);
                    let ok = std::thread::Builder::new()
                        .name("genai-account".into())
                        .spawn(move || {
                            let r = genai::relay::Relay::new(&base, tok)
                                .me(&genai::http::Http::new())
                                .map(|a| (a.account, a.balance_usd))
                                .map_err(|e| format!("{e:#}"));
                            let _ = tx.send(r);
                        })
                        .is_ok();
                    self.jobs.genai.account = if ok { Account::Fetching(rx) } else { Account::Known(Err("no network here".into())) };
                }
                Account::Fetching(rx) => {
                    if let Ok(r) = rx.try_recv() {
                        self.jobs.genai.account = Account::Known(r);
                    }
                }
                Account::Known(_) => {}
            }
        }
        let Some(r) = self.jobs.genai.run.as_mut() else { return };
        let node = r.node;
        // A capture whose bake died (the input stopped rendering, the rig
        // was rebuilt) leaves no bake and nothing captured.
        if matches!(r.phase, Phase::Capturing { .. })
            && self.graph_bake.is_none()
            && self.jobs.genai.captured.is_empty()
            && self.jobs.genai.clip_frames.is_empty()
        {
            let why = self.load_errors.get(&node).cloned().unwrap_or_else(|| "the input did not render".into());
            r.phase = Phase::Failed(why);
            return;
        }
        let mut landed = None;
        match &mut r.phase {
            Phase::Quoting { rx, .. } => {
                if let Ok(q) = rx.try_recv() {
                    let Phase::Quoting { plan, inputs, .. } = std::mem::replace(&mut r.phase, Phase::Failed(String::new())) else {
                        unreachable!()
                    };
                    r.phase = Phase::Ready { plan, inputs, quote: q };
                }
            }
            Phase::Running { job } => {
                let mut end = None;
                while let Ok(ev) = job.rx.try_recv() {
                    match ev {
                        Event::Step(i, s) => {
                            r.steps.insert(i, s);
                        }
                        Event::Done(t) => end = Some(Ok(t)),
                        Event::Failed(e) => end = Some(Err(e)),
                    }
                }
                match end {
                    Some(Ok(t)) => {
                        let spent = t.usd.map(|u| format!(", {}", service::usd(u))).unwrap_or_default();
                        r.phase = Phase::Done(format!("take landed{spent}"));
                        landed = Some(t);
                    }
                    Some(Err(e)) => r.phase = Phase::Failed(e),
                    None => {}
                }
            }
            _ => {}
        }
        if let Some(t) = landed {
            // An undo step, on purpose: one generation, one user action.
            // Running an unchanged pipeline again is a cache hit with the
            // same key: pick the take it already made, never a duplicate.
            if let Some(p) = self.egraph.node_mut(node).and_then(|n| params_mut(&mut n.kind)) {
                let pl = &mut p.pipeline;
                match pl.takes.iter().position(|x| x.key == t.key && x.file == t.file) {
                    Some(k) => pl.take = k,
                    None => {
                        pl.takes.push(t);
                        pl.take = pl.takes.len() - 1;
                    }
                }
            }
            self.genai_show(node);
            // The relay balance moved.
            if matches!(self.jobs.genai.cfg.route, Route::Relay) {
                self.jobs.genai.account = Account::Unknown;
            }
        }
    }

    /// The text on a GenAI node's prompt port: the String node wired there,
    /// through any dots on the way.
    /// The pipeline a run of `node` would walk: the node's own, except that
    /// a one-model node with a String wired into its prompt port reads the
    /// port instead of its typed prompt. Never written back to the node;
    /// staleness compares against this, not the stored steps.
    pub(crate) fn genai_runnable(&self, node: u64) -> Option<Pipeline> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let mut pl = p.pipeline.clone();
        if p.model.is_some() && self.genai_wired_prompt(node).is_some() {
            for s in &mut pl.steps {
                if matches!(s.op, Op::Prompt { .. }) {
                    s.op = Op::PromptIn;
                }
            }
        }
        Some(pl)
    }

    /// A one-model node upstream of `node`'s image or matte that has no take
    /// yet: it would pass its own input through, and the capture would send
    /// the raw plate instead of that model's result. Its name, if there is
    /// one. (Bypassed nodes pass through on purpose and are skipped.)
    fn genai_unrun_upstream(&self, node: u64) -> Option<String> {
        let mut todo: Vec<u64> = [0, 1].iter().filter_map(|&i| self.egraph.input_of(node, i)).collect();
        let mut seen = std::collections::BTreeSet::new();
        while let Some(n) = todo.pop() {
            if !seen.insert(n) {
                continue;
            }
            let Some(en) = self.egraph.node(n) else { continue };
            if let NodeKind::GenAi(p) = &en.kind {
                if !en.disabled && p.pipeline.picked().is_none() {
                    return Some(crate::editor::display_label(en));
                }
                if p.pipeline.picked().is_some() {
                    continue; // its take is what renders; above it is not read
                }
            }
            todo.extend(self.egraph.edges.iter().filter(|e| e.0 == n).map(|e| e.2));
        }
        None
    }

    pub(crate) fn genai_wired_prompt(&self, node: u64) -> Option<String> {
        wired_prompt(&self.egraph, node)
    }

    /// The view a step inside a GenAI Pipeline's dive shows: its owner's,
    /// with the owner's wiring read off the graph the dive parked.
    pub(crate) fn genai_step_view(&self) -> View {
        let Some((owner, g)) = self
            .dive
            .last()
            .filter(|d| d.kind == crate::DiveKind::GenAi)
            .and_then(|d| Some((d.from?, &d.graph)))
        else {
            return self.genai_view(0);
        };
        View {
            has_input: g.input_of(owner, 0).is_some(),
            has_matte: g.input_of(owner, 1).is_some(),
            prompt: wired_prompt(g, owner),
            ..self.genai_view(owner)
        }
    }

    /// Has the wired prompt changed since the quote was made? Then the
    /// quote is for a different request.
    fn genai_prompt_moved(&self, node: u64, pl: &Pipeline, inputs: &Inputs) -> bool {
        pl.wants_prompt() && self.genai_wired_prompt(node) != inputs.prompt
    }

    /// Point the node's reader at its picked take (or at nothing).
    pub(crate) fn genai_show(&mut self, node: u64) {
        let want = self
            .egraph
            .node(node)
            .and_then(|n| params(&n.kind))
            .and_then(|p| p.pipeline.picked())
            .map(|t| t.path(&config::results_dir()));
        // A video take in a build that cannot decode one is not loaded: the
        // panel says why, and the node passes its input through.
        let want = want.filter(|p| !genai::pipeline::is_video_file(&p.to_string_lossy()) || video_ok());
        match want {
            Some(path) if path.exists() => {
                let s = path.to_string_lossy().into_owned();
                if self.loaded.get(&node).map(|l| l.path.as_str()) != Some(s.as_str()) {
                    self.load_read(node, &s);
                    self.playback_dirty = true;
                }
            }
            _ => {
                if self.loaded.remove(&node).is_some() {
                    self.playback_dirty = true;
                }
            }
        }
    }

    /// Every GenAI node shows its picked take: after a setup load, an undo,
    /// a paste — anything that swapped the graph under the readers.
    pub(crate) fn genai_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n.kind, NodeKind::GenAi(_)))
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.genai_show(id);
        }
    }
}

// ------------------------------------------------------------ --test-genai

/// A stand-in fal on localhost for `--test-genai`: the queue dance, a
/// pricing row, and a flat sRGB-128 grey card as the result. Counts the
/// submits, which is how the suite proves nothing was sent before "pay &
/// run" and nothing at all on a cached re-run.
pub mod fake {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    pub struct Fal {
        pub url: String,
        pub submits: Arc<AtomicUsize>,
    }

    impl Fal {
        pub fn submits(&self) -> usize {
            self.submits.load(Ordering::SeqCst)
        }
    }

    /// The grey card's code value, and what it decodes to.
    pub const GREY: u8 = 128;

    pub fn start() -> std::io::Result<Fal> {
        let l = TcpListener::bind("127.0.0.1:0")?;
        let url = format!("http://127.0.0.1:{}", l.local_addr()?.port());
        let submits = Arc::new(AtomicUsize::new(0));
        let (base, n) = (url.clone(), submits.clone());
        let lin = genai::image::srgb_decode(GREY as f32 / 255.0);
        let png = genai::image::png_srgb_rgb(&vec![lin; 64 * 36 * 4], 64, 36).unwrap_or_default();
        std::thread::spawn(move || {
            for c in l.incoming().flatten() {
                let (base, n, png) = (base.clone(), n.clone(), png.clone());
                std::thread::spawn(move || {
                    let mut conn = c;
                    let mut r = BufReader::new(conn.try_clone().unwrap());
                    let mut first = String::new();
                    let _ = r.read_line(&mut first);
                    let mut len = 0usize;
                    loop {
                        let mut line = String::new();
                        if r.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                            break;
                        }
                        if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                            len = v.trim().parse().unwrap_or(0);
                        }
                    }
                    let mut body = vec![0u8; len];
                    let _ = r.read_exact(&mut body);
                    let mut it = first.split_whitespace();
                    let (m, p) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
                    let json = |v: serde_json::Value| ("application/json", serde_json::to_vec(&v).unwrap());
                    let (ct, out) = match (m, p) {
                        ("POST", "/fal-ai/flux/schnell") => {
                            n.fetch_add(1, Ordering::SeqCst);
                            json(serde_json::json!({
                                "request_id": "r1",
                                "status_url": format!("{base}/fal-ai/flux/requests/r1/status"),
                                "response_url": format!("{base}/fal-ai/flux/requests/r1"),
                                "cancel_url": format!("{base}/fal-ai/flux/requests/r1/cancel"),
                                "queue_position": 0,
                            }))
                        }
                        ("GET", p) if p.ends_with("/status") => json(serde_json::json!({
                            "status": "COMPLETED", "request_id": "r1",
                            "response_url": format!("{base}/fal-ai/flux/requests/r1"),
                            "logs": [], "metrics": {"inference_time": 0.4},
                        })),
                        ("GET", "/fal-ai/flux/requests/r1") => json(serde_json::json!({
                            "images": [{"url": format!("{base}/cdn/grey.png")}],
                            "has_nsfw_concepts": [false],
                        })),
                        ("GET", "/cdn/grey.png") => ("image/png", png),
                        ("GET", p) if p.starts_with("/v1/models/pricing") => json(serde_json::json!({
                            "prices": [{"endpoint_id": "fal-ai/flux/schnell", "unit_price": 0.003,
                                        "unit": "megapixels", "currency": "USD"}],
                            "next_cursor": null, "has_more": false,
                        })),
                        _ => json(serde_json::json!({"detail": "no route"})),
                    };
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        out.len()
                    );
                    let _ = conn.write_all(head.as_bytes());
                    let _ = conn.write_all(&out);
                });
            }
        });
        Ok(Fal { url, submits })
    }
}

/// `--test-genai`'s state.
pub struct TestGenai {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    /// The free-form pipeline node, once a one-model node takes `node`.
    pipe: u64,
    /// The plate through the node before anything is generated.
    before: Option<f32>,
    /// A probe waiting for a second one that agrees.
    last: Option<f32>,
    fake: Option<fake::Fal>,
}

impl Default for TestGenai {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, pipe: 0, before: None, last: None, fake: None }
    }
}

impl Gfx {
    /// `--test-genai`: a GenAI node end to end in the live app, offline.
    ///
    /// Local: Reference -> Output. Nothing is billed, so a click runs it
    /// straight through: the input is captured by a one-frame bake at the
    /// playhead, shrunk, sent through sRGB into the results directory, the
    /// take lands as the node's reader, and the picture through the node is
    /// the plate again (within the 8-bit round trip; above 1.0 clips).
    ///
    /// Remote, against `fake::Fal`: Prompt -> FLUX schnell -> Output on
    /// "my fal key". The click stops at a QUOTE from the pricing API with
    /// nothing submitted; "pay & run" submits exactly once; the grey card
    /// the server returned is what the node shows, in the working space;
    /// and a second run of the unchanged pipeline is a cache hit that sends
    /// nothing and adds no take. Waits are wall-clock.
    pub(crate) fn drive_test_genai(&mut self) {
        let Some(mut t) = self.test.genai.take() else { return };
        let fail = |m: String| -> ! {
            println!("GENAI TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 120 {
            fail(format!("timeout at stage {} (run: {})", t.stage, self.genai_state_words()));
        }
        let wait = |t: &TestGenai, ms: u64| t.since.elapsed().as_millis() as u64 >= ms;
        // A reading two probes agree on, 150 ms apart, with the playhead
        // landed: the rig rebuild and the upload have both happened.
        let stable = |me: &mut Self, t: &mut TestGenai| -> Option<f32> {
            if me.force_display || me.eval_tick() != me.timeline_pos || !wait(t, 150) {
                return None;
            }
            let v = me.probe_output_box(900, 500, 32, 32, 1)?;
            t.since = web_time::Instant::now();
            match t.last.replace(v) {
                Some(p) if (p - v).abs() < 1e-5 => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        let next = |t: &mut TestGenai, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        let pipeline = |me: &mut Self, node: u64, f: &dyn Fn(&mut Pipeline)| {
            if let Some(p) = me.egraph.node_mut(node).and_then(|n| params_mut(&mut n.kind)) {
                f(&mut p.pipeline);
            }
        };
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                // Hermetic: our own results dir and config, a stand-in fal,
                // and my-key billing — never the user's files or account.
                let dir = std::env::temp_dir().join(format!("wobble-test-genai-{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir);
                let fake = fake::start().unwrap_or_else(|e| fail(format!("fake fal: {e}")));
                std::env::set_var("WOBBLE_GENAI_DIR", dir.join("results"));
                std::env::set_var("WOBBLE_GENAI_CONFIG", dir.join("genai.json"));
                std::env::set_var("WOBBLE_FAL_URL", &fake.url);
                std::env::set_var("FAL_KEY", "test-key");
                std::env::set_var("WOBBLE_GENAI_ROUTE", "own");
                std::env::remove_var("WOBBLE_RELAY_TOKEN");
                self.jobs.genai = Runs::default();
                t.fake = Some(fake);
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let mut p = GenAiParams::new(slot);
                // Reference -> Output: nothing remote.
                p.pipeline = Pipeline { steps: vec![], takes: vec![], take: 0 };
                let r = p.pipeline.push(Op::Reference, [0.0, -0.5]);
                let o = p.pipeline.push(Op::Output, [0.0, 0.5]);
                p.pipeline.connect(r, o, 0);
                let g = self.egraph.add(NodeKind::GenAi(p), egui::pos2(360.0, 200.0));
                self.egraph.connect(g, 0, read);
                self.egraph.view = Some(g);
                // The transport runs under test: hold one frame, or the
                // plate changes under the probe (and under the capture).
                self.playing = false;
                t.node = g;
                next(&mut t, 1);
            }
            1 => {
                if let Some(v) = stable(self, &mut t) {
                    t.before = Some(v);
                    self.genai_action(t.node, Action::Prepare);
                    next(&mut t, 2);
                }
            }
            2 => match self.jobs.genai.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the local run failed: {e}")),
                Some(Phase::Done(_)) => {
                    let takes = self.egraph.node(t.node).and_then(|n| params(&n.kind)).map(|p| p.pipeline.takes.len());
                    if takes != Some(1) {
                        fail(format!("expected one take, have {takes:?}"));
                    }
                    if !self.loaded.contains_key(&t.node) {
                        fail("the take did not load into the node's reader".into());
                    }
                    next(&mut t, 3);
                }
                _ => {}
            },
            3 => {
                if let Some(v) = stable(self, &mut t) {
                    let want = t.before.unwrap().clamp(0.0, 1.0);
                    if (v - want).abs() > 0.02 + 0.03 * want {
                        fail(format!("the round trip moved the picture: plate {want:.4}, take {v:.4}"));
                    }
                    // Now the remote pipeline.
                    self.genai_action(t.node, Action::Dismiss);
                    // The prompt comes off a Prompt node on the text pipe.
                    pipeline(self, t.node, &|pl| {
                        pl.steps = Pipeline::default().steps;
                        pl.steps[0].op = Op::PromptIn;
                    });
                    let text = NodeKind::Prompt(engine::graph::PromptParams { text: "an 18% grey card".into() });
                    let pn = self.egraph.add(text, egui::pos2(520.0, 120.0));
                    self.egraph.connect(t.node, 2, pn);
                    self.genai_action(t.node, Action::Prepare);
                    next(&mut t, 4);
                }
            }
            4 => match self.jobs.genai.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the remote prepare failed: {e}")),
                Some(Phase::Ready { quote, inputs, .. }) => {
                    if inputs.prompt.as_deref() != Some("an 18% grey card") {
                        fail(format!("the wired Prompt node did not reach the run: {:?}", inputs.prompt));
                    }
                    let submits = t.fake.as_ref().unwrap().submits();
                    if submits != 0 {
                        fail(format!("{submits} request(s) submitted before pay & run"));
                    }
                    let total = quote.as_ref().ok().and_then(|q| q.total());
                    if total.is_none_or(|u| (u - 0.003).abs() > 1e-9) {
                        fail(format!("the quote is {quote:?}, expected $0.003 for one megapixel"));
                    }
                    self.genai_action(t.node, Action::Confirm);
                    next(&mut t, 5);
                }
                _ => {}
            },
            5 => match self.jobs.genai.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the remote run failed: {e}")),
                Some(Phase::Done(_)) => {
                    let submits = t.fake.as_ref().unwrap().submits();
                    if submits != 1 {
                        fail(format!("pay & run submitted {submits} requests, expected one"));
                    }
                    next(&mut t, 6);
                }
                _ => {}
            },
            6 => {
                if let Some(v) = stable(self, &mut t) {
                    let want = genai::image::srgb_decode(fake::GREY as f32 / 255.0);
                    if (v - want).abs() > 0.01 {
                        fail(format!("the grey card reads {v:.4} through the node, expected {want:.4}"));
                    }
                    if engine::cost::licence_of(&self.egraph.node(t.node).unwrap().kind)
                        != engine::cost::Licence::Permissive
                    {
                        fail("a schnell take is tagged non-commercial".into());
                    }
                    self.genai_action(t.node, Action::Dismiss);
                    self.genai_action(t.node, Action::Prepare);
                    next(&mut t, 7);
                }
            }
            7 => match self.jobs.genai.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the cached re-run failed: {e}")),
                Some(Phase::Ready { .. }) => fail("an unchanged pipeline asked to be paid for again".into()),
                Some(Phase::Done(_)) => {
                    let submits = t.fake.as_ref().unwrap().submits();
                    let takes = self.egraph.node(t.node).and_then(|n| params(&n.kind)).map(|p| p.pipeline.takes.len());
                    if submits != 1 {
                        fail(format!("the cached re-run sent {} request(s)", submits - 1));
                    }
                    if takes != Some(2) {
                        fail(format!("the cached re-run changed the takes: {takes:?}, expected 2"));
                    }
                    // One-model nodes, made the way the add menu makes them.
                    self.genai_action(t.node, Action::Dismiss);
                    let entry = |label: &str| {
                        crate::editor::node_catalog()
                            .into_iter()
                            .find(|e| e.label == label)
                            .unwrap_or_else(|| fail(format!("no \"{label}\" in the add menu")))
                    };
                    let mut slot = || {
                        self.next_read_slot += 1;
                        self.next_read_slot - 1
                    };
                    let (s1, s2) = (slot(), slot());
                    let flux = (entry("FLUX.1 [schnell]").make)(s1);
                    let kontext = (entry("FLUX.1 Kontext [pro]").make)(s2);
                    if crate::editor::node_label(&flux) != "FLUX.1 [schnell]" || innards::provider(&flux).is_some() {
                        fail("a one-model node is not called by its model, or still has steps inside".into());
                    }
                    let m = self.egraph.add(flux, egui::pos2(360.0, 360.0));
                    let k = self.egraph.add(kontext, egui::pos2(360.0, 480.0));
                    let pn = self.egraph.input_of(t.node, 2).unwrap();
                    self.egraph.connect(m, 2, pn);
                    self.egraph.connect(k, 0, m);
                    self.egraph.connect(k, 2, pn);
                    // A chain whose upstream model never ran must not send
                    // the raw plate in its place.
                    self.genai_action(k, Action::Prepare);
                    match self.jobs.genai.run.as_ref().map(|r| &r.phase) {
                        Some(Phase::Failed(e)) if e.contains("no take yet") => {}
                        _ => fail(format!(
                            "a model downstream of an un-run one was let through ({})",
                            self.genai_state_words()
                        )),
                    }
                    self.genai_action(k, Action::Dismiss);
                    // The String on the port, the same model, seed and size:
                    // the same request as the pipeline's, so a cache hit.
                    self.genai_action(m, Action::Prepare);
                    t.pipe = t.node;
                    t.node = m;
                    next(&mut t, 8);
                }
                _ => {}
            },
            8 => match self.jobs.genai.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the one-model node's run failed: {e}")),
                Some(Phase::Ready { .. }) => fail("a one-model node re-billed a request the pipeline already made".into()),
                Some(Phase::Done(_)) => {
                    let submits = t.fake.as_ref().unwrap().submits();
                    let takes = self.egraph.node(t.node).and_then(|n| params(&n.kind)).map(|p| p.pipeline.takes.len());
                    if submits != 1 || takes != Some(1) {
                        fail(format!("the one-model node sent {} request(s) and has {takes:?} take(s)", submits - 1));
                    }
                    // The pipeline opens like a Noise: its steps are nodes.
                    let before = self.egraph.node(t.pipe).and_then(|n| params(&n.kind)).cloned().unwrap();
                    self.dive_into(t.pipe).unwrap_or_else(|e| fail(format!("the pipeline would not open: {e}")));
                    let steps = self.egraph.nodes.values().filter(|n| matches!(n.kind, NodeKind::GenAiStep(_))).count();
                    if steps != before.pipeline.steps.len() || !self.dive.last().is_some_and(|d| d.kind == crate::DiveKind::GenAi) {
                        fail(format!("the dive shows {steps} step node(s) for {} steps", before.pipeline.steps.len()));
                    }
                    let input = crate::editor::node_catalog()
                        .into_iter()
                        .find(|e| e.cat == crate::editor::NodeCat::GenAiStep && e.label == "Input")
                        .unwrap_or_else(|| fail("no Input step in the dive's menu".into()));
                    let step = self.egraph.add((input.make)(0), egui::pos2(0.0, -400.0));
                    // Viewing the Input step shows what the node's image
                    // input carries, not the node's take.
                    let saved = self.egraph.view.replace(step);
                    let shown = self.render_preview(self.timeline_pos).and_then(|g| g.view);
                    let wired = self.dive.last().and_then(|d| d.graph.input_of(t.pipe, 0));
                    if shown.is_none() || shown != wired {
                        fail(format!("viewing the Input step shows {shown:?}, not the node's input {wired:?}"));
                    }
                    self.egraph.view = saved;
                    self.dive_out().unwrap_or_else(|e| fail(format!("the pipeline would not close: {e}")));
                    let after = self.egraph.node(t.pipe).and_then(|n| params(&n.kind)).cloned().unwrap();
                    if after.pipeline.steps.len() != before.pipeline.steps.len() + 1
                        || !matches!(after.pipeline.steps.last().map(|s| &s.op), Some(Op::Reference))
                    {
                        fail("a step added inside did not come back out as a step".into());
                    }
                    if after.pipeline.takes != before.pipeline.takes {
                        fail("opening and closing the pipeline changed its takes".into());
                    }
                    if clip_ok() {
                        // A clip model: the plate as the clip AND the matte
                        // (any picture is a matte), half a second of it.
                        let entry = crate::editor::node_catalog()
                            .into_iter()
                            .find(|e| e.label == "LTX 2.3 video inpaint")
                            .unwrap_or_else(|| fail("no LTX 2.3 video inpaint in the add menu".into()));
                        self.next_read_slot += 1;
                        let mut kind = (entry.make)(self.next_read_slot - 1);
                        if let Some(p) = params_mut(&mut kind) {
                            for s in &mut p.pipeline.steps {
                                if let Op::ClipIn { seconds } = &mut s.op {
                                    *seconds = 0.5;
                                }
                                if let Op::Prompt { text } = &mut s.op {
                                    *text = "empty street".into();
                                }
                            }
                        }
                        let read = self.egraph.input_of(t.pipe, 0).unwrap();
                        let v = self.egraph.add(kind, egui::pos2(360.0, 600.0));
                        self.egraph.connect(v, 0, read);
                        self.egraph.connect(v, 1, read);
                        self.timeline_pos = 0;
                        self.genai_action(v, Action::Prepare);
                        t.node = v;
                        next(&mut t, 9);
                        self.test.genai = Some(t);
                        return;
                    }
                    println!(
                        "GENAI TEST PASS: a local pipeline captured the plate at the playhead and \
                         played it back through the node within the 8-bit round trip; a fal \
                         pipeline prompted by a Prompt node on the text pipe stopped at a $0.003 quote with nothing sent, submitted exactly \
                         once on pay & run, showed the returned grey card in the working space, \
                         and a re-run of the unchanged pipeline was a cache hit that sent nothing; \
                         one-model nodes from the add menu are named for their model, refuse to \
                         run below an un-run model, and read a wired String as the same request; \
                         the pipeline opens as a graph of step nodes and a step added there comes \
                         back out as a step"
                    );
                    std::process::exit(0);
                }
                _ => {}
            },
            9 => match self.jobs.genai.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the clip run failed before the quote: {e}")),
                Some(Phase::Ready { inputs, .. }) => {
                    let (Some(c), Some(m)) = (&inputs.clip, &inputs.matte_clip) else {
                        fail("the clip run captured no clip or no matte clip".into())
                    };
                    let want = (0.5 * self.fps).round() as u32;
                    if c.frames != want || m.frames != want || (c.w, c.h) != (m.w, m.h) {
                        fail(format!("clip {}x{} x{}, matte {}x{} x{}, expected {want} frames each, same size",
                            c.w, c.h, c.frames, m.w, m.h, m.frames));
                    }
                    if c.w.min(c.h) > CLIP_SHORT || c.w % 2 != 0 || c.h % 2 != 0 {
                        fail(format!("the clip is {}x{}: over {CLIP_SHORT} px short side, or odd", c.w, c.h));
                    }
                    #[cfg(feature = "movie")]
                    {
                        let f = std::env::temp_dir().join(format!("wobble-test-genai-clip-{}.mp4", std::process::id()));
                        let _ = std::fs::write(&f, &c.bytes);
                        let n = engine::decode::ExrDecoder::frame_count(&engine::movie::MovieDecoder::default(), &f);
                        let _ = std::fs::remove_file(&f);
                        if n.as_ref().ok() != Some(&(want as u64)) {
                            fail(format!("the clip's mp4 decodes to {n:?} frames, expected {want}"));
                        }
                    }
                    let submits = t.fake.as_ref().unwrap().submits();
                    if submits != 1 {
                        fail(format!("the clip run submitted {} request(s) before pay & run", submits - 1));
                    }
                    println!(
                        "GENAI TEST PASS: local, remote, cached, one-model and dive stages as before; \
                         an LTX video inpaint node captured {want} frames of the plate and of its \
                         matte from the playhead, encoded them as a {}x{} mp4 that decodes back \
                         to {want} frames, and stopped at the quote with nothing sent",
                        c.w, c.h
                    );
                    std::process::exit(0);
                }
                _ => {}
            },
            _ => {}
        }
        self.test.genai = Some(t);
    }

    fn genai_state_words(&self) -> String {
        match self.jobs.genai.run.as_ref().map(|r| &r.phase) {
            None => "none".into(),
            Some(Phase::Capturing { todo, .. }) => format!("capturing {todo:?}, bake {}", self.graph_bake.is_some()),
            Some(Phase::Quoting { .. }) => "quoting".into(),
            Some(Phase::Ready { .. }) => "ready".into(),
            Some(Phase::Running { .. }) => "running".into(),
            Some(Phase::Failed(e)) => format!("failed: {e}"),
            Some(Phase::Done(s)) => format!("done: {s}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn kind() -> NodeKind {
        NodeKind::GenAi(GenAiParams::new(0))
    }

    #[test]
    fn a_2k_clip_frame_comes_out_at_720_and_even() {
        let px = vec![0.18f32; 2048 * 1080 * 4];
        let (rgb, w, h) = super::clip_frame(&px, 2048, 1080, true);
        assert_eq!((w, h), (1364, 720));
        assert_eq!(rgb.len(), (w * h * 3) as usize);
        let want = (genai::image::srgb_encode(0.18) * 255.0 + 0.5) as u8;
        assert!(rgb.iter().all(|&v| v == want));
        // Smaller than the cap: kept, cut to even.
        let (_, w, h) = super::clip_frame(&vec![0.5; 513 * 289 * 4], 513, 289, false);
        assert_eq!((w, h), (512, 288));
    }

    #[test]
    fn the_dive_round_trips_the_pipeline() {
        let NodeKind::GenAi(p) = kind() else { unreachable!() };
        let g = to_editor(&p.pipeline);
        // prompt, generate, output, two wires, spread out of the face.
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 2);
        assert!(g.nodes.values().any(|n| n.pos.x.abs() > 1.5 || n.pos.y.abs() > 1.5));
        let back = from_editor(&g);
        assert_eq!(back.len(), 3);
        for (a, b) in back.iter().zip(&p.pipeline.steps) {
            assert_eq!((&a.op, &a.inputs), (&b.op, &b.inputs));
        }
        // Graph positions survive a second trip untouched.
        let mut pl = p.pipeline.clone();
        pl.steps = back.clone();
        assert_eq!(from_editor(&to_editor(&pl)), back);
    }

    #[test]
    fn coming_out_drops_a_wire_of_the_wrong_pipe() {
        let NodeKind::GenAi(p) = kind() else { unreachable!() };
        let mut g = to_editor(&p.pipeline);
        // Ids are step + 1: the prompt's text into Output's image port.
        g.edges.retain(|e| e.0 != 3);
        g.edges.push((3, 0, 1));
        let back = from_editor(&g);
        assert_eq!(back[2].inputs, vec![None]);
    }

    #[test]
    fn every_step_kind_is_in_the_dive_menu_and_no_model_twice() {
        let e = step_entries();
        for want in ["Prompt", "Prompt In", "Input", "Matte", "ComfyUI", "Output"] {
            assert!(e.iter().any(|(l, _)| *l == want), "{want}");
        }
        let mut labels: Vec<_> = e.iter().map(|(l, _)| *l).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), e.len());
    }

    fn view(state: ViewState) -> View {
        View {
            route: Route::Relay,
            bill: Ok("wobble credits · tester · $12.50".into()),
            fal: None,
            relay: Some(Origin::File),
            comfy_url: config::COMFY_URL.into(),
            state,
            steps: BTreeMap::new(),
            busy_elsewhere: false,
            has_input: true,
            has_matte: false,
            prompt: None,
            take_missing: true,
            fps: 24.0,
        }
    }

    /// The panel must FIT its tiles (CLAUDE.md: a row pushed into the
    /// tile's scroll area cannot be clicked). Measured at its worst: the
    /// tallest step (a text-to-image model tagged non-commercial, with its
    /// size slider and extra field), a quote waiting to be paid, a
    /// missing-take warning and five takes — with the keys section shut,
    /// and again with it OPEN, because opening it must not push "pay & run"
    /// off the tile.
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        // The free-form pipeline, and the busiest one-model node (prompt,
        // image AND matte inputs).
        for (keys_open, model) in [(false, None), (true, None), (false, Some("fal/gpt-image-2-inpaint"))] {
            let mut p = match model {
                Some(id) => GenAiParams::for_model(0, id),
                None => GenAiParams::new(0),
            };
            if model.is_none() {
                p.pipeline.set_model(1, "fal/flux-dev");
            }
            for k in 0..5 {
                p.pipeline.takes.push(genai::Take {
                    key: format!("k{k}"),
                    file: "result.png".into(),
                    tick: 1001,
                    commercial: k % 2 == 0,
                    usd: Some(0.025),
                    label: "flux dev: a moth on a lamp at night, rain".into(),
                    when: 0,
                    data: false,
                });
            }
            let quote = Ok(Quote {
                items: vec![Some(service::Price { usd: 0.025, exact: true, note: "per image".into() })],
                balance_usd: Some(12.5),
            });
            let v = view(ViewState::Ready { quote, billed: 1, stale: true });
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                // Select the model step, and open (or not) the keys.
                innards::select(ui.ctx(), 1, 1);
                let id = egui::Id::new(("genai-keys", 1u64));
                let mut st = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
                st.set_open(keys_open);
                st.store(ui.ctx());
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let _ = panel(ui, 1, &mut p, &v);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(match (keys_open, model) {
                (_, Some(_)) => "genai-panel-model",
                (true, _) => "genai-panel-keys",
                _ => "genai-panel",
            });
            assert!(
                over.get() <= 0.5,
                "the panel (keys open: {keys_open}) overflows its tiles by {}pt",
                over.get()
            );
            assert!(wide.get() <= 0.5, "the panel overflows sideways by {}pt", wide.get());
        }
    }

    /// The panel as it really draws, text and all, to
    /// `target/genai-panel-gpu.png`: `cargo test -p app genai_panel_render
    /// -- --ignored`.
    #[test]
    #[ignore]
    fn genai_panel_render() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let mut p = GenAiParams::new(0);
        p.pipeline.set_model(1, "fal/flux-dev");
        if let Op::Prompt { text } = &mut p.pipeline.steps[0].op {
            *text = "a moth on a lamp at night, rain on the glass".into();
        }
        for (k, c) in [(0, true), (1, false)] {
            p.pipeline.takes.push(genai::Take {
                key: format!("k{k}"),
                file: "result.png".into(),
                tick: 1001,
                commercial: c,
                usd: Some(0.025),
                label: "flux dev: a moth on a lamp".into(),
                when: 0,
                data: false,
            });
        }
        let quote = Ok(Quote {
            items: vec![Some(service::Price { usd: 0.025, exact: true, note: "per image".into() })],
            balance_usd: Some(12.5),
        });
        let v = view(ViewState::Ready { quote, billed: 1, stale: false });
        let out = crate::uishot::render("genai-panel-gpu", span(TILES), span(TILE_ROWS), 2.0, 4, |ui, _| {
            innards::select(ui.ctx(), 1, 1);
            let _ = panel(ui, 1, &mut p, &v);
        });
        match out {
            Some(path) => println!("wrote {}", path.display()),
            None => println!("SKIP: no GPU adapter"),
        }
    }
}
