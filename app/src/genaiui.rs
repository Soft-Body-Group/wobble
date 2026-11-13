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
use genai::run::{self, Event, Inputs, Plan, Png, StepState};
use genai::service::{self, Quote};
use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;

/// Panel footprint: two tiles wide (steps | run), two tall.
pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;
/// Largest reference we send, in megapixels (see `genai::image::shrink`).
pub const REF_MP: f32 = 2.0;
/// Inner graphs stay legible at this many steps.
pub const MAX_STEPS: usize = 12;

/// Can this build play a video take? Only a build that decodes movies
/// (`--features app/movie`) offers a video step: one it cannot play back
/// would bill for a clip and show nothing (the `SUPPORTED_EXTS` rule).
pub fn video_ok() -> bool {
    engine::decode::is_supported(std::path::Path::new("take.mp4"))
}

/// Is this model one this build may offer?
fn offered(m: &catalog::RemoteModel) -> bool {
    !m.task.is_video() || video_ok()
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

// ------------------------------------------------------------ inner graph

/// The pipeline as an inner graph: one node per step, every step an ITEM
/// (selecting it puts its fields in the panel).
pub struct Steps;
pub static STEPS: Steps = Steps;

impl innards::Provider for Steps {
    fn mini(&self, kind: &NodeKind) -> innards::Mini {
        use innards::{Mini, MiniEdge, MiniNode};
        let Some(p) = params(kind) else { return Mini::default() };
        let steps = &p.pipeline.steps;
        let nodes = steps
            .iter()
            .enumerate()
            .map(|(i, s)| MiniNode {
                id: i,
                at: s.at,
                label: s.op.label(),
                inputs: s.op.ports().len(),
                output: s.op.output().is_some(),
                item: true,
                removable: !matches!(s.op, Op::Output),
            })
            .collect();
        let edges = steps
            .iter()
            .enumerate()
            .flat_map(|(to, s)| {
                s.inputs
                    .iter()
                    .enumerate()
                    .filter_map(move |(port, src)| src.map(|from| MiniEdge { from, to, port }))
            })
            .collect();
        Mini { nodes, edges }
    }

    fn can_add(&self, kind: &NodeKind) -> bool {
        params(kind).is_some_and(|p| p.pipeline.steps.len() < MAX_STEPS)
    }

    /// `+` adds a model step — the thing a pipeline most often grows by.
    /// The panel's "+ step" menu adds the rest.
    fn add(&self, graph: &mut crate::editor::EditorGraph, node: u64) -> Option<usize> {
        let p = params_mut(&mut graph.node_mut(node)?.kind)?;
        let op = Op::Generate { model: catalog::DEFAULT_MODEL.into(), seed: 1, opts: Opts::default() };
        Some(push_near(&mut p.pipeline, op))
    }

    fn remove(&self, graph: &mut crate::editor::EditorGraph, node: u64, id: usize) -> bool {
        graph
            .node_mut(node)
            .and_then(|n| params_mut(&mut n.kind))
            .is_some_and(|p| p.pipeline.remove(id))
    }

    fn move_to(&self, kind: &mut NodeKind, id: usize, at: [f32; 2]) {
        if let Some(s) = params_mut(kind).and_then(|p| p.pipeline.steps.get_mut(id)) {
            s.at = at;
        }
    }

    fn connect(&self, kind: &mut NodeKind, from: usize, to: usize, port: usize) -> bool {
        params_mut(kind).is_some_and(|p| p.pipeline.connect(from, to, port))
    }

    fn disconnect(&self, kind: &mut NodeKind, to: usize, port: usize) {
        if let Some(p) = params_mut(kind) {
            p.pipeline.disconnect(to, port);
        }
    }
}

/// Add a step somewhere free-ish: along a row above Output, so a new step
/// never lands on top of the last one.
fn push_near(pl: &mut Pipeline, op: Op) -> usize {
    let n = pl.steps.len() as f32;
    let x: f32 = -0.6 + 0.3 * (n % 5.0);
    let y: f32 = -0.4 + 0.2 * ((n / 5.0).floor() % 3.0);
    let at = [x.clamp(-0.85, 0.85), y.clamp(-0.85, 0.85)];
    pl.push(op, at)
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
    account: Account,
}

#[cfg_attr(not(feature = "commerce"), allow(dead_code))]
enum Account {
    Unknown,
    Fetching(Receiver<Result<(String, f64), String>>),
    Known(Result<(String, f64), String>),
}

impl Default for Runs {
    fn default() -> Self {
        Self { cfg: Config::load(), run: None, captured: vec![], account: Account::Unknown }
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
    let n = p.pipeline.steps.len();
    let sel = innards::selected(ui.ctx(), node, n);
    ui.columns(2, |cols| {
        steps_column(&mut cols[0], node, p, sel, v);
        if let Some(a) = run_column(&mut cols[1], node, p, v) {
            act = Some(a);
        }
    });
    act
}

fn steps_column(ui: &mut Ui, node: u64, p: &mut GenAiParams, sel: usize, v: &View) {
    let pl = &mut p.pipeline;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 3.0;
        for i in 0..pl.steps.len() {
            let label = pl.steps[i].op.label();
            if ui.add(egui::Button::selectable(i == sel, small(label))).clicked() {
                innards::select(ui.ctx(), node, i);
            }
        }
        ui.menu_button(small("+ step"), |ui| {
            let mut add: Option<Op> = None;
            if ui.button("prompt").clicked() {
                add = Some(Op::Prompt { text: String::new() });
            }
            if ui.button("prompt in (the node's prompt port)").clicked() {
                add = Some(Op::PromptIn);
            }
            if ui.button("input (the node's image)").clicked() {
                add = Some(Op::Reference);
            }
            if ui.button("matte (the node's matte)").clicked() {
                add = Some(Op::Matte);
            }
            ui.separator();
            for t in Task::ALL {
                if let Some(m) = catalog::for_task(t).find(|m| offered(m)) {
                    if ui.button(t.label()).clicked() {
                        add = Some(Op::Generate { model: m.id.into(), seed: 1, opts: Opts::default() });
                    }
                }
            }
            ui.separator();
            if ui.button("ComfyUI workflow").clicked() {
                add = Some(Op::Comfy { workflow: String::new(), seed: 1, commercial: false });
            }
            if let Some(op) = add {
                if pl.steps.len() < MAX_STEPS {
                    let i = push_near(pl, op);
                    innards::select(ui.ctx(), node, i);
                }
                ui.close();
            }
        });
    });
    ui.add_space(4.0);
    let Some(is_output) = pl.steps.get(sel).map(|s| matches!(s.op, Op::Output)) else { return };
    let mut remove = false;
    let mut model_change: Option<&'static str> = None;
    let id = |what: &str| egui::Id::new(("genai", node, sel, what.to_string()));
    match &mut pl.steps[sel].op {
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
        Op::Output => {
            ui.label(small("the node's picture: wire the step whose result you want"));
            ui.weak(small("stretched over the node's input format, so an edit lines up with its plate"));
        }
        Op::Generate { model, seed, opts } => {
            let cur = catalog::get(model);
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
    if !is_output {
        ui.add_space(4.0);
        if ui.small_button("remove step").clicked() {
            remove = true;
        }
    }
    if let Some(m) = model_change {
        pl.set_model(sel, m);
    }
    if remove {
        pl.remove(sel);
    }
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
        let cfg = &self.genai.cfg;
        let bill = match cfg.route {
            #[cfg(not(feature = "commerce"))]
            Route::Relay => Err("not available in this build".to_string()),
            #[cfg(feature = "commerce")]
            Route::Relay => match (&cfg.relay_token, &self.genai.account) {
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
        let mine = self.genai.run.as_ref().filter(|r| r.node == node);
        let state = match mine.map(|r| &r.phase) {
            None => ViewState::Idle,
            Some(Phase::Capturing { .. }) => ViewState::Capturing,
            Some(Phase::Quoting { .. }) => ViewState::Quoting,
            Some(Phase::Ready { plan, quote, inputs }) => {
                let r = mine.unwrap();
                let stale = self
                    .egraph
                    .node(node)
                    .and_then(|n| params(&n.kind))
                    .is_none_or(|p| p.pipeline.steps != r.pipeline.steps)
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
            busy_elsewhere: self.genai.run.as_ref().is_some_and(|r| r.node != node && r.active()),
            has_input: self.egraph.input_of(node, 0).is_some(),
            has_matte: self.egraph.input_of(node, 1).is_some(),
            prompt: self.genai_wired_prompt(node),
            take_missing,
        }
    }

    pub(crate) fn genai_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Prepare => self.genai_prepare(node),
            Action::Confirm => self.genai_confirm(node),
            Action::Cancel => {
                if let Some(r) = self.genai.run.take() {
                    if let Phase::Running { job } = &r.phase {
                        job.cancel();
                    }
                    if matches!(r.phase, Phase::Capturing { .. }) {
                        // The capture is a one-frame bake: drop it.
                        self.graph_bake = None;
                        self.genai.captured.clear();
                    }
                    // A running job's thread sees the cancel and cancels
                    // the remote request; nobody is listening for its end.
                }
            }
            Action::Dismiss => {
                if self.genai.run.as_ref().is_some_and(|r| !r.active()) {
                    self.genai.run = None;
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
            Action::Refresh => self.genai.account = Account::Unknown,
        }
    }

    fn genai_reload_config(&mut self) {
        self.genai.cfg = Config::load();
        self.genai.account = Account::Unknown;
    }

    fn genai_fail(&mut self, node: u64, pipeline: Pipeline, msg: impl Into<String>) {
        self.genai.run = Some(Run {
            node,
            pipeline,
            tick: self.timeline_pos as i64,
            phase: Phase::Failed(msg.into()),
            steps: BTreeMap::new(),
        });
    }

    fn genai_prepare(&mut self, node: u64) {
        if self.genai.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else { return };
        let pl = p.pipeline;
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
            if let Err(e) = self.genai.cfg.can_bill() {
                return self.genai_fail(node, pl, e);
            }
        }
        let prompt = self.genai_wired_prompt(node);
        if pl.wants_prompt() && prompt.is_none() {
            return self.genai_fail(node, pl, "the pipeline reads the node's prompt: wire a String node into it");
        }
        let prompt = prompt.filter(|_| pl.wants_prompt());
        let mut todo = vec![];
        for (port, wants, what) in [(0, pl.wants_reference(), "image"), (1, pl.wants_matte(), "matte")] {
            if wants {
                if self.egraph.input_of(node, port).is_none() {
                    return self.genai_fail(node, pl, format!("the pipeline reads the node's {what}: wire one in"));
                }
                todo.push(port);
            }
        }
        if !todo.is_empty() && self.graph_bake.is_some() {
            return self.genai_fail(node, pl, "a bake is running: wait for it, then run");
        }
        let Some(comp) = self.comp.output_size() else {
            return self.genai_fail(node, pl, "nothing is rendering yet");
        };
        let tick = self.timeline_pos as i64;
        self.genai.captured.clear();
        self.genai.run = Some(Run {
            node,
            pipeline: pl,
            tick,
            phase: Phase::Capturing { inputs: Inputs { reference: None, matte: None, prompt, comp, tick }, todo },
            steps: BTreeMap::new(),
        });
        self.genai_capture_next();
    }

    /// Render the next input the run needs, or move on to the plan.
    fn genai_capture_next(&mut self) {
        let Some(r) = &self.genai.run else { return };
        let Phase::Capturing { todo, .. } = &r.phase else { return };
        let (node, tick, next, pl) = (r.node, r.tick, todo.first().copied(), r.pipeline.clone());
        match next {
            Some(port) => {
                let Some(input) = self.egraph.input_of(node, port) else {
                    return self.genai_fail(node, pl, "an input was unwired mid-run");
                };
                if let Err(e) = self.start_capture(node, input, port, tick) {
                    self.genai_fail(node, pl, e);
                }
            }
            None => self.genai_plan(),
        }
    }

    /// A one-frame bake of `input` at `tick`, whose pixels land in
    /// `genai.captured` instead of a file (`BakeSink::Capture`).
    fn start_capture(&mut self, node: u64, input: u64, port: usize, tick: i64) -> Result<(), String> {
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
            len: tick + 1,
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
        let got = std::mem::take(&mut self.genai.captured);
        let Some(r) = self.genai.run.as_mut().filter(|r| r.node == node) else { return };
        let Phase::Capturing { inputs, todo } = &mut r.phase else { return };
        let Some(port) = todo.first().copied() else { return };
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
        let Some(r) = self.genai.run.as_mut() else { return };
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
                run::Services::from_config(&self.genai.cfg),
                dir,
                vec![],
            );
            r.phase = Phase::Running { job };
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (pl, pl2, ins) = (r.pipeline.clone(), plan.clone(), inputs.clone());
        let cfg = self.genai.cfg.clone();
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
        let current = self.egraph.node(node).and_then(|n| params(&n.kind)).map(|p| p.pipeline.steps.clone());
        let moved = match self.genai.run.as_ref().filter(|r| r.node == node) {
            Some(Run { pipeline, phase: Phase::Ready { inputs, .. }, .. }) => {
                self.genai_prompt_moved(node, pipeline, inputs)
            }
            _ => false,
        };
        let Some(r) = self.genai.run.as_mut().filter(|r| r.node == node) else { return };
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
            run::Services::from_config(&self.genai.cfg),
            config::results_dir(),
            approved,
        );
        r.phase = Phase::Running { job };
    }

    /// Once a redraw: quotes landing, run progress, the account balance.
    pub(crate) fn genai_poll(&mut self) {
        // The relay balance, fetched once per config change.
        #[cfg(feature = "commerce")]
        if self.genai.cfg.route == Route::Relay && self.genai.cfg.relay_token.is_some() {
            match &self.genai.account {
                Account::Unknown => {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let (base, tok) = (self.genai.cfg.relay_url.clone(), self.genai.cfg.relay_token.clone().unwrap().0);
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
                    self.genai.account = if ok { Account::Fetching(rx) } else { Account::Known(Err("no network here".into())) };
                }
                Account::Fetching(rx) => {
                    if let Ok(r) = rx.try_recv() {
                        self.genai.account = Account::Known(r);
                    }
                }
                Account::Known(_) => {}
            }
        }
        let Some(r) = self.genai.run.as_mut() else { return };
        let node = r.node;
        // A capture whose bake died (the input stopped rendering, the rig
        // was rebuilt) leaves no bake and nothing captured.
        if matches!(r.phase, Phase::Capturing { .. }) && self.graph_bake.is_none() && self.genai.captured.is_empty() {
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
            if matches!(self.genai.cfg.route, Route::Relay) {
                self.genai.account = Account::Unknown;
            }
        }
    }

    /// The text on a GenAI node's prompt port: the String node wired there,
    /// through any dots on the way.
    pub(crate) fn genai_wired_prompt(&self, node: u64) -> Option<String> {
        let mut src = self.egraph.input_of(node, 2)?;
        for _ in 0..64 {
            match &self.egraph.node(src)?.kind {
                NodeKind::Prompt(p) => return Some(p.text.clone()),
                NodeKind::Dot { .. } => src = self.egraph.input_of(src, 0)?,
                _ => return None,
            }
        }
        None
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
    /// The plate through the node before anything is generated.
    before: Option<f32>,
    /// A probe waiting for a second one that agrees.
    last: Option<f32>,
    fake: Option<fake::Fal>,
}

impl Default for TestGenai {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, before: None, last: None, fake: None }
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
                self.genai = Runs::default();
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
            2 => match self.genai.run.as_ref().map(|r| &r.phase) {
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
            4 => match self.genai.run.as_ref().map(|r| &r.phase) {
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
            5 => match self.genai.run.as_ref().map(|r| &r.phase) {
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
            7 => match self.genai.run.as_ref().map(|r| &r.phase) {
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
                    println!(
                        "GENAI TEST PASS: a local pipeline captured the plate at the playhead and \
                         played it back through the node within the 8-bit round trip; a fal \
                         pipeline prompted by a Prompt node on the text pipe stopped at a $0.003 quote with nothing sent, submitted exactly \
                         once on pay & run, showed the returned grey card in the working space, \
                         and a re-run of the unchanged pipeline was a cache hit that sent nothing"
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
        match self.genai.run.as_ref().map(|r| &r.phase) {
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
    use innards::Provider;

    fn kind() -> NodeKind {
        NodeKind::GenAi(GenAiParams::new(0))
    }

    #[test]
    fn the_inner_graph_mirrors_the_pipeline() {
        let k = kind();
        let m = STEPS.mini(&k);
        // prompt, generate, output
        assert_eq!(m.nodes.len(), 3);
        assert_eq!(m.edges.len(), 2);
        let out = m.nodes.iter().find(|n| n.label == "out").unwrap();
        assert!(!out.removable && !out.output && out.inputs == 1);
        assert!(m.edge_into(out.id, 0).is_some());
    }

    #[test]
    fn inner_wiring_goes_through_the_pipeline_s_rule() {
        let mut k = kind();
        // The prompt's text into Output's image port is refused.
        assert!(!STEPS.connect(&mut k, 0, 2, 0));
        STEPS.disconnect(&mut k, 2, 0);
        assert!(STEPS.mini(&k).edge_into(2, 0).is_none());
        assert!(STEPS.connect(&mut k, 1, 2, 0));
    }

    #[test]
    fn add_and_remove_through_the_editor_graph() {
        let mut g = crate::editor::EditorGraph::default();
        let n = g.add(kind(), Pos2::ZERO);
        let i = STEPS.add(&mut g, n).unwrap();
        assert_eq!(i, 3);
        assert!(STEPS.remove(&mut g, n, 0), "the prompt goes");
        // Output is not removable.
        let out = params(&g.node(n).unwrap().kind).unwrap().pipeline.output().unwrap();
        assert!(!STEPS.remove(&mut g, n, out));
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
        for keys_open in [false, true] {
            let mut p = GenAiParams::new(0);
            p.pipeline.set_model(1, "fal/flux-dev");
            for k in 0..5 {
                p.pipeline.takes.push(genai::Take {
                    key: format!("k{k}"),
                    file: "result.png".into(),
                    tick: 1001,
                    commercial: k % 2 == 0,
                    usd: Some(0.025),
                    label: "flux dev: a moth on a lamp at night, rain".into(),
                    when: 0,
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
            shot.save(if keys_open { "genai-panel-keys" } else { "genai-panel" });
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

    #[test]
    fn a_new_step_never_lands_on_the_last_one() {
        let mut p = Pipeline::default();
        let a = push_near(&mut p, Op::Reference);
        let b = push_near(&mut p, Op::Matte);
        assert_ne!(p.steps[a].at, p.steps[b].at);
    }
}
