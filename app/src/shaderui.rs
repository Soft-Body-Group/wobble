//! UI for the Shader node: a code window over a live comp, plus the
//! auto-generated parameter rows in the properties tile.
//!
//! The contract that makes typing WGSL at a running compositor bearable:
//!
//! * The text you are editing is a BUFFER, not the node. Diagnostics run on
//!   every keystroke (`usershader::compile` is memoized), but nothing reaches
//!   the graph until you apply — because applying is a topology change, and
//!   a topology change is a pipeline build.
//! * A shader that does not compile is not an error state to be rescued
//!   from: the node passes its input through, the comp keeps playing, and
//!   the errors sit under the editor until you fix them.
//! * Param VALUES are uniforms and survive an apply by name, so tuning a
//!   slider is free and renaming nothing keeps everything.

use crate::editor::{EditorGraph, PDst, PSrc};
use crate::theme;
use crate::tools::{self, ControlKind};
use crate::widgets::{self, HudSlider, RowCtx};
use engine::graph::{NodeKind, ShaderParams};
use engine::usershader::{self, ParamTy};
use std::collections::{BTreeMap, HashMap};

/// Per-node edit buffers, and which node's window is open.
#[derive(Default)]
pub struct Editors {
    bufs: HashMap<u64, String>,
    pub open: Option<u64>,
}

impl Editors {
    fn buf(&mut self, node: u64, src: &str) -> &mut String {
        self.bufs.entry(node).or_insert_with(|| src.to_string())
    }

    /// Is there an edit buffer for this node that differs from what the
    /// graph is actually rendering?
    pub fn dirty(&self, node: u64, src: &str) -> bool {
        self.bufs.get(&node).is_some_and(|b| b != src)
    }

    pub fn retain(&mut self, live: impl Fn(u64) -> bool) {
        self.bufs.retain(|id, _| live(*id));
        if self.open.is_some_and(|n| !live(n)) {
            self.open = None;
        }
    }
}

/// What a Shader row wants the app to do once the node's borrow is gone.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Req {
    /// Open the code window on this node.
    Edit(u64),
    /// Give every declared param a wired control node (see [`make_controls`]).
    MakeControls(u64),
}

/// One control node, ready to be added: its kind, caption, the lane values
/// to author on it, and which of the shader's channels each lane drives.
struct Plan {
    kind: ControlKind,
    label: String,
    /// (control lane name, value) — `value` for a slider, `r`/`g`/`b` for rgb.
    lanes: Vec<(&'static str, f32)>,
    lo: f32,
    hi: f32,
    /// Parallel to `lanes`: the Shader channel each one drives.
    targets: Vec<crate::anim::Channel>,
}

/// Every control a shader's params want, in declaration order.
///
/// The mapping is deliberately plain: a colour is one RGB control (three
/// lanes, three cables), a vec2 is TWO sliders because there is no
/// two-lane instrument and pretending a colour widget is one would be a
/// lie, and a bool is a 0..1 slider because the control set has no toggle.
/// Where that last one starts to grate, the answer is a `ui.toggle` control
/// kind in `tools.rs`, not a special case here.
fn plans(def: &usershader::ShaderDef, values: &BTreeMap<String, [f32; 4]>) -> Vec<Plan> {
    let mut out = Vec::new();
    for spec in &def.params {
        let v = values.get(&spec.name).copied().unwrap_or(spec.default);
        let chan = |lane: usize| {
            crate::anim::Channel::Shader(crate::anim::intern(&crate::anim::shader_key(
                &spec.name, spec.ty, lane,
            )))
        };
        match spec.ty {
            ParamTy::Color => out.push(Plan {
                kind: ControlKind::Rgb,
                label: spec.label.clone(),
                lanes: vec![("r", v[0]), ("g", v[1]), ("b", v[2])],
                lo: spec.min,
                hi: spec.max,
                targets: (0..3).map(chan).collect(),
            }),
            ParamTy::Vec2 => {
                for (i, axis) in ["x", "y"].into_iter().enumerate() {
                    out.push(Plan {
                        kind: ControlKind::Slider,
                        label: format!("{} {axis}", spec.label),
                        lanes: vec![("value", v[i])],
                        lo: spec.min,
                        hi: spec.max,
                        targets: vec![chan(i)],
                    });
                }
            }
            // A bool has no instrument of its own yet; 0..1 is honest about
            // that rather than inventing a half-toggle.
            ParamTy::Bool => out.push(Plan {
                kind: ControlKind::Slider,
                label: spec.label.clone(),
                lanes: vec![("value", v[0])],
                lo: 0.0,
                hi: 1.0,
                targets: vec![chan(0)],
            }),
            ParamTy::Float | ParamTy::Int => out.push(Plan {
                kind: ControlKind::Slider,
                label: spec.label.clone(),
                lanes: vec![("value", v[0])],
                lo: spec.min,
                hi: spec.max,
                targets: vec![chan(0)],
            }),
        }
    }
    out
}

/// Give every declared param of a Shader node its own control node, wired
/// in. Returns how many controls were made.
///
/// This is the bridge to the panel composer: a control node IS a knob on the
/// tool the body becomes, so a shader wrapped in a tool arrives with a real
/// panel instead of a list of channels somebody has to hand-cable. Writing
/// the shader is the interesting part; wiring six sliders is not.
///
/// Idempotent — a param already driven by a cable is left alone, so pressing
/// the button again after adding a param tops up rather than duplicating.
/// Controls are spawned UNPLACED and handed to `tools::place_controls`, so
/// the composer's own layout rules decide where they land.
pub fn make_controls(graph: &mut EditorGraph, node: u64, cols: i32, rows: i32) -> usize {
    let Some(NodeKind::Shader(p)) = graph.nodes.get(&node).map(|n| &n.kind) else {
        return 0;
    };
    let plans = plans(&p.def(), &p.values);
    let base = graph.nodes.get(&node).map(|n| n.pos).unwrap_or_default();
    let mut made = 0;
    for plan in plans {
        // Already cabled? Then this param has a knob (or something better)
        // and is none of our business.
        if plan
            .targets
            .iter()
            .all(|t| graph.patch.link_to(PDst::Chan(node, *t)).is_some())
        {
            continue;
        }
        let mut params: BTreeMap<String, f32> = BTreeMap::new();
        for (lane, v) in &plan.lanes {
            params.insert((*lane).to_string(), *v);
        }
        params.insert("lo".into(), plan.lo);
        params.insert("hi".into(), plan.hi);
        let pos = base + egui::vec2(-240.0, made as f32 * 64.0 - 40.0);
        let id = graph.add(
            NodeKind::Custom { type_id: plan.kind.type_id().into(), params },
            pos,
        );
        if let Some(n) = graph.nodes.get_mut(&id) {
            n.ui_label = plan.label.clone();
        }
        for ((lane, _), target) in plan.lanes.iter().zip(&plan.targets) {
            graph.patch.link(
                PSrc::Chan(id, crate::anim::Channel::Custom(crate::anim::intern(lane))),
                PDst::Chan(node, *target),
            );
        }
        made += 1;
    }
    if made > 0 {
        tools::place_controls(graph, cols, rows);
    }
    made
}

/// The properties-tile half: generated rows for every declared param, the
/// shader's status, and the way into the code window.
/// `dirty` and `req` are passed in and out rather than the whole `Editors`
/// and the graph: the caller is holding a mutable borrow of the node at this
/// point, so anything that touches either is recorded and applied after.
pub fn params_ui(
    ui: &mut egui::Ui,
    cx: &mut RowCtx,
    node: u64,
    params: &mut ShaderParams,
    dirty: bool,
    req: &mut Option<Req>,
) -> Option<HudSlider> {
    let def = params.def();
    let mut hud = None;

    ui.horizontal(|ui| {
        if ui.button("edit shader…").clicked() {
            *req = Some(Req::Edit(node));
        }
        if !def.params.is_empty()
            && ui
                .button("make controls")
                .on_hover_text(
                    "one control node per param, wired in.\n\
                     inside a subwob they become its knobs (arrange them in\n\
                     the panel composer); in a plain comp they are sliders in\n\
                     the graph. params already cabled are left alone.",
                )
                .clicked()
        {
            *req = Some(Req::MakeControls(node));
        }
        if dirty {
            ui.colored_label(theme::warn(), "edited");
        }
    });

    if !def.ok() {
        // Conspicuous, and it says what the node is DOING, not just that
        // something is wrong: the picture on screen is the input, unchanged.
        ui.colored_label(
            theme::error(),
            format!("{} error(s) — passing through", def.errors.len()),
        );
        if let Some(e) = def.errors.first() {
            ui.weak(e.to_string());
        }
        return hud;
    }

    for spec in &def.params {
        let chan = |lane: usize| {
            crate::anim::Channel::Shader(crate::anim::intern(&crate::anim::shader_key(
                &spec.name, spec.ty, lane,
            )))
        };
        let mut v = params.values.get(&spec.name).copied().unwrap_or(spec.default);
        let h = match spec.ty {
            ParamTy::Bool => {
                let c = [chan(0)];
                let mut on = v[0] > 0.5;
                ui.horizontal(|ui| {
                    ui.label(&spec.label);
                    ui.checkbox(&mut on, "");
                    widgets::key_dot(ui, cx, &c);
                });
                v[0] = on as u32 as f32;
                None
            }
            ParamTy::Int => {
                let c = [chan(0)];
                let mut f = v[0];
                let h = widgets::slider(ui, cx, &spec.label, &mut f, spec.min, spec.max, &c);
                v[0] = f.round();
                h
            }
            ParamTy::Float => {
                let c = [chan(0)];
                let mut f = v[0];
                let h = if spec.min < spec.max {
                    widgets::slider(ui, cx, &spec.label, &mut f, spec.min, spec.max, &c)
                } else {
                    widgets::drags(ui, cx, &spec.label, &mut [&mut f], 0.01, (0.0, 1.0), &c)
                };
                v[0] = f;
                h
            }
            ParamTy::Vec2 | ParamTy::Color => {
                let n = spec.ty.lanes();
                let chans: Vec<_> = (0..n).map(chan).collect();
                let (mut a, mut b, mut c2) = (v[0], v[1], v[2]);
                let mut refs: Vec<&mut f32> = vec![&mut a, &mut b];
                if n == 3 {
                    refs.push(&mut c2);
                }
                let h = widgets::drags(
                    ui,
                    cx,
                    &spec.label,
                    &mut refs,
                    0.005,
                    (spec.min, spec.max),
                    &chans,
                );
                drop(refs);
                v[0] = a;
                v[1] = b;
                if n == 3 {
                    v[2] = c2;
                }
                h
            }
        };
        params.values.insert(spec.name.clone(), v);
        if h.is_some() {
            hud = h;
        }
    }
    ui.weak(format!(
        "{} · {} in · {} pass{}",
        def.label,
        def.inputs.len(),
        def.passes,
        if def.passes == 1 { "" } else { "es" }
    ));
    // Rule 8: a slow path is never quiet. Every input costs a full-canvas
    // resolve, every pass costs a full-canvas texture, and the node fuses
    // with nothing — the panel says so rather than letting it be a surprise
    // at 4K.
    ui.colored_label(
        theme::cost_note(),
        "SLOW PATH: own passes, no fusing.",
    );
    ui.weak(format!(
        "{} resolve pass{} + {} full-canvas result{}.",
        def.inputs.len(),
        if def.inputs.len() == 1 { "" } else { "es" },
        def.passes,
        if def.passes == 1 { "" } else { "s" }
    ));
    hud
}

/// The floating code window. Returns true when the source was applied (the
/// caller snapshots for undo and lets the topology hash do the rest).
pub fn window(
    ctx: &egui::Context,
    node: u64,
    title: &str,
    params: &mut ShaderParams,
    editors: &mut Editors,
) -> bool {
    let mut open = true;
    let mut applied = false;
    // Compiled from the BUFFER, not the node: diagnostics are about what you
    // are typing, not about what is currently rendering.
    let buf_src = editors.buf(node, &params.source).clone();
    let def = usershader::compile(&buf_src);
    let dirty = buf_src != params.source;

    egui::Window::new(format!("{title} — shader"))
        .id(egui::Id::new(("shader-editor", node)))
        .open(&mut open)
        .resizable(true)
        .default_size([720.0, 560.0])
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                let can = dirty && def.ok();
                if ui
                    .add_enabled(can, egui::Button::new("apply  ⌘↩"))
                    .on_hover_text("recompiles the node")
                    .clicked()
                {
                    applied = true;
                }
                if ui.add_enabled(dirty, egui::Button::new("revert")).clicked() {
                    editors.bufs.insert(node, params.source.clone());
                }
                ui.menu_button("examples", |ui| {
                    for (name, src) in EXAMPLES {
                        if ui.button(*name).clicked() {
                            editors.bufs.insert(node, src.to_string());
                            ui.close();
                        }
                    }
                });
                // Files are for SHARING. The source of truth is the setup —
                // a shader travels with the comp, which is the one thing a
                // Matchbox cannot do.
                if ui.button("open…").on_hover_text("load a .wgsl into the buffer").clicked() {
                    if let Some(path) = crate::filedialog::FileDialog::new()
                        .add_filter("wgsl shader", &["wgsl", "txt"])
                        .pick_file()
                    {
                        match std::fs::read_to_string(&path) {
                            Ok(t) => {
                                editors.bufs.insert(node, t);
                            }
                            Err(e) => log::error!("shader open {}: {e}", path.display()),
                        }
                    }
                }
                if ui.button("save as…").clicked() {
                    if let Some(path) = crate::filedialog::FileDialog::new()
                        .add_filter("wgsl shader", &["wgsl"])
                        .set_file_name("shader.wgsl")
                        .save_file()
                    {
                        if let Err(e) = std::fs::write(&path, &buf_src) {
                            log::error!("shader save {}: {e}", path.display());
                        }
                    }
                }
                ui.separator();
                if def.ok() {
                    ui.colored_label(
                        theme::ok(),
                        format!(
                            "ok · {} in · {} pass{}",
                            def.inputs.len(),
                            def.passes,
                            if def.passes == 1 { "" } else { "es" }
                        ),
                    );
                } else {
                    ui.colored_label(
                        theme::error(),
                        format!("{} error(s)", def.errors.len()),
                    );
                }
                if dirty && def.ok() {
                    ui.weak("unapplied");
                }
            });

            // ⌘↵ applies. Checked at window scope so it works wherever the
            // caret is — a code editor that needs you to reach for a button
            // is a code editor nobody iterates in.
            if dirty
                && def.ok()
                && ui.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Enter))
            {
                applied = true;
            }

            let errs_h = if def.ok() { 0.0 } else { 120.0 };
            let code_h = (ui.available_height() - errs_h - 40.0).max(120.0);
            egui::ScrollArea::vertical()
                .id_salt("shader-code")
                .max_height(code_h)
                .show(ui, |ui| {
                    let buf = editors.buf(node, &params.source);
                    ui.add(
                        egui::TextEdit::multiline(buf)
                            .code_editor()
                            .desired_width(f32::INFINITY)
                            .desired_rows(24),
                    );
                });

            if !def.ok() {
                ui.separator();
                egui::ScrollArea::vertical()
                    .id_salt("shader-errs")
                    .max_height(errs_h)
                    .show(ui, |ui| {
                        for e in &def.errors {
                            ui.colored_label(
                                theme::error(),
                                e.to_string(),
                            );
                        }
                    });
            }

            ui.separator();
            ui.collapsing("reference", |ui| {
                ui.label(egui::RichText::new(REFERENCE).monospace().size(11.0));
            });
        });

    if applied {
        // The graph's own undo watcher snapshots on the next mouse-up, so an
        // apply is undoable without this doing anything about it.
        let src = editors.buf(node, &params.source).clone();
        params.set_source(src);
    }
    if !open {
        editors.open = None;
    }
    applied
}

const REFERENCE: &str = "\
pragmas (//! lines, anywhere in the file)
  name: <text>                      node label
  input: <ident>                    an image port (up to 4; first is port 0)
  passes: <1..4>                    how many passes to run
  group: <text>                     groups the params that follow
  param: <ty> <name> = <v> [lo, hi] \"Label\"
        ty = float | int | bool | vec2 | color

entry points
  fn pass1(uv: vec2f, frag: vec2f) -> vec4f     (or main_image, 1 pass only)
  fn pass2(...)  ... one per declared pass

bindings
  <input name>      the wired input, resolved to a texture
  result1..resultN  an earlier pass's output (pass i sees 1..i-1)
  P.<param>         your params, by name and WGSL type
  P.res  P.time  P.frame  P.pass_index  P.aspect  P.luma_w

helpers
  tex(t, uv)        bilinear, clamp to edge   (Matchbox texture2D)
  tex_b(t, uv)      bilinear, black outside the frame
  tex_px(t, p)      bilinear at PIXEL coords
  texel(t, ivec)    exact texel, clamped
  tex_size(t)       texture size in pixels
  luma(rgb)         luminance in the comp's WORKING primaries

notes
  the comp is SCENE-LINEAR (ACEScg): 1.0 is not white, values go above it
  params are uniforms — moving one never recompiles; editing text does
  bool is u32 (`if P.flip != 0u`), int is i32";

/// Shipped examples. Each one exists to show a thing the fused plugin path
/// cannot do, so a first read of any of them teaches the difference.
pub const EXAMPLES: &[(&str, &str)] = &[
    ("Radial blur (arbitrary sampling)", EX_RADIAL),
    ("Box blur (two passes)", EX_BOX2),
    ("Plasma (generator, no input)", EX_PLASMA),
    ("Chromatic aberration", EX_CHROMA),
];

const EX_RADIAL: &str = r#"//! name: Radial Blur
//! input: front
//! param: float amount = 0.15 [0, 1] "Amount"
//! param: vec2 centre = (0.5, 0.5) [0, 1] "Centre"
//! param: int steps = 24 [2, 64] "Steps"

// The whole point of a Shader node: sampling the input somewhere OTHER than
// this pixel, in a loop. A fused plugin expression cannot do this.
fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    let dir = uv - P.centre;
    var sum = vec4f(0.0);
    let n = max(P.steps, 1);
    for (var i = 0; i < n; i = i + 1) {
        let t = f32(i) / f32(n);
        sum = sum + tex(front, uv - dir * t * P.amount);
    }
    return sum / f32(n);
}
"#;

const EX_BOX2: &str = r#"//! name: Box Blur
//! input: front
//! passes: 2
//! param: float radius = 8.0 [0, 64] "Radius"

// Separable: pass 1 blurs across, pass 2 blurs down what pass 1 wrote.
// `result1` is pass 1's own full-canvas texture.
fn blur(t: texture_2d<f32>, frag: vec2f, dir: vec2f) -> vec4f {
    let r = i32(clamp(P.radius, 0.0, 64.0));
    var sum = vec4f(0.0);
    var n = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
        sum = sum + tex_px(t, frag + dir * f32(i));
        n = n + 1.0;
    }
    return sum / max(n, 1.0);
}
fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return blur(front, frag, vec2f(1.0, 0.0)); }
fn pass2(uv: vec2f, frag: vec2f) -> vec4f { return blur(result1, frag, vec2f(0.0, 1.0)); }
"#;

const EX_PLASMA: &str = r#"//! name: Plasma
//! param: float scale = 6.0 [1, 40] "Scale"
//! param: float speed = 1.0 [0, 4] "Speed"
//! param: color tint = (1.0, 0.6, 0.3) "Tint"

// A generator: nothing wired in, `uv` and `P.time` do all the work — the
// shape most Shadertoy shaders take. The comp is scene-linear, so the
// result is deliberately kept near 0..1 rather than pushed to display white.
fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    let p = (uv - vec2f(0.5)) * vec2f(P.aspect, 1.0) * P.scale;
    let t = P.time * P.speed;
    let v = sin(p.x + t) + sin(p.y + t * 0.7)
          + sin(length(p) * 1.5 - t * 1.3);
    let f = v * 0.25 + 0.5;
    return vec4f(P.tint * f, 1.0);
}
"#;

const EX_CHROMA: &str = r#"//! name: Chromatic Aberration
//! input: front
//! param: float amount = 4.0 [0, 40] "Amount"
//! param: bool radial = true "Radial"

// Three different sample positions for three channels — pointwise nodes
// cannot express this at all, because there is no single sample.
fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    var dir = vec2f(1.0, 0.0);
    if (P.radial != 0u) { dir = normalize(uv - vec2f(0.5) + vec2f(1e-6)); }
    let d = dir * P.amount;
    let r = tex_px(front, frag + d).r;
    let g = tex_px(front, frag).g;
    let b = tex_px(front, frag - d).b;
    let a = tex_px(front, frag).a;
    return vec4f(r, g, b, a);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_example_compiles() {
        for (name, src) in EXAMPLES {
            let d = usershader::compile(src);
            assert!(d.ok(), "{name}: {:?}", d.errors);
        }
    }

    #[test]
    fn examples_cover_the_shapes_they_claim() {
        assert_eq!(usershader::compile(EX_BOX2).passes, 2);
        // The plasma is a generator: it declares no input, so it gets the
        // one implicit port and never samples it.
        assert!(!EX_PLASMA.contains("//! input:"));
        assert!(usershader::compile(EX_PLASMA).ok());
    }

    #[test]
    fn buffers_are_forgotten_with_their_nodes() {
        let mut e = Editors::default();
        e.buf(7, "x");
        e.open = Some(7);
        e.retain(|id| id != 7);
        assert!(e.bufs.is_empty());
        assert_eq!(e.open, None);
    }

    const MIXED: &str = r#"//! input: front
//! param: float gain = 1.0 [0, 4] "Gain"
//! param: color tint = (1, 1, 1) "Tint"
//! param: vec2 centre = (0.5, 0.5) [0, 1] "Centre"
//! param: bool invert = false "Invert"
fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    let c = tex(front, uv) * P.gain;
    return vec4f(select(c.rgb, 1.0 - c.rgb, P.invert != 0u) * P.tint, c.a);
}
"#;

    fn shader_graph(src: &str) -> (EditorGraph, u64) {
        let mut g = EditorGraph::default();
        let mut p = ShaderParams::default();
        p.set_source(src.to_string());
        let id = g.add(NodeKind::Shader(p), egui::Pos2::ZERO);
        (g, id)
    }

    fn driven(g: &EditorGraph, node: u64, name: &str) -> bool {
        let ch = crate::anim::Channel::Shader(crate::anim::intern(name));
        g.patch.link_to(PDst::Chan(node, ch)).is_some()
    }

    #[test]
    fn make_controls_covers_every_param_type() {
        let (mut g, id) = shader_graph(MIXED);
        // float 1 + colour 1 + vec2 2 + bool 1
        assert_eq!(make_controls(&mut g, id, tools::cols(2), tools::rows(1)), 5);

        // Scalars drive the bare name; vectors drive one lane each.
        assert!(driven(&g, id, "gain"));
        assert!(driven(&g, id, "invert"));
        for l in 0..3 {
            assert!(driven(&g, id, &format!("tint.{l}")), "tint lane {l}");
        }
        for l in 0..2 {
            assert!(driven(&g, id, &format!("centre.{l}")), "centre lane {l}");
        }

        // One RGB control for the colour, sliders for the rest.
        let kinds: Vec<_> = g
            .nodes
            .values()
            .filter_map(|n| tools::control_of(&n.kind))
            .collect();
        assert_eq!(kinds.iter().filter(|k| **k == ControlKind::Rgb).count(), 1);
        assert_eq!(kinds.iter().filter(|k| **k == ControlKind::Slider).count(), 4);

        // Every control is PLACED, so the panel is usable straight away.
        assert!(g
            .nodes
            .values()
            .filter(|n| tools::control_of(&n.kind).is_some())
            .all(|n| n.ui_cell.is_some()));
    }

    #[test]
    fn make_controls_is_idempotent() {
        let (mut g, id) = shader_graph(MIXED);
        let first = make_controls(&mut g, id, tools::cols(2), tools::rows(1));
        assert_eq!(make_controls(&mut g, id, tools::cols(2), tools::rows(1)), 0);
        assert_eq!(
            g.nodes.values().filter(|n| tools::control_of(&n.kind).is_some()).count(),
            first
        );
    }

    #[test]
    fn a_new_param_tops_up_rather_than_duplicating() {
        let (mut g, id) = shader_graph(MIXED);
        make_controls(&mut g, id, tools::cols(2), tools::rows(1));
        // Add a param to the shader, the way an edit would.
        if let Some(NodeKind::Shader(p)) = g.nodes.get_mut(&id).map(|n| &mut n.kind) {
            p.set_source(format!("{MIXED}//! param: float extra = 2.0 [0, 5]\n"));
        }
        assert_eq!(make_controls(&mut g, id, tools::cols(2), tools::rows(1)), 1);
        assert!(driven(&g, id, "extra"));
    }

    #[test]
    fn a_control_starts_where_the_shader_actually_is() {
        // Not the authored default: pressing the button must not move the
        // picture.
        let (mut g, id) = shader_graph(MIXED);
        if let Some(NodeKind::Shader(p)) = g.nodes.get_mut(&id).map(|n| &mut n.kind) {
            p.values.insert("gain".into(), [3.5, 0.0, 0.0, 0.0]);
        }
        make_controls(&mut g, id, tools::cols(2), tools::rows(1));
        let gain_ch = crate::anim::Channel::Shader(crate::anim::intern("gain"));
        let src = g.patch.link_to(PDst::Chan(id, gain_ch)).unwrap().from;
        let PSrc::Chan(ctrl, _) = src else { panic!("not a channel source") };
        let NodeKind::Custom { params, .. } = &g.nodes[&ctrl].kind else {
            panic!("not a control")
        };
        assert_eq!(params.get("value"), Some(&3.5));
        assert_eq!(params.get("hi"), Some(&4.0), "range came from the pragma");
    }

    /// The whole story in one test: write a shader, press "make controls",
    /// save the body as a tool, and the instance has one knob per shader
    /// param that really reaches the shader. If this passes, the node is
    /// wrapped by the panel composer with nothing hand-wired.
    #[test]
    fn a_shader_plus_make_controls_is_a_tool_with_knobs() {

        let (mut body, sh) = shader_graph(MIXED);
        let inp = body.add(
            NodeKind::Custom { type_id: crate::tools::INPUT_IMAGE.into(), params: Default::default() },
            egui::Pos2::ZERO,
        );
        body.edges.push((sh, 0, inp));
        body.view = Some(sh);
        assert_eq!(make_controls(&mut body, sh, tools::cols(2), tools::rows(1)), 5);

        tools::register(tools::finish(
            "t.shaderui".into(),
            "ShaderUI".into(),
            [0; 3],
            1,
            1,
            body,
        ));
        let def = tools::get("t.shaderui").expect("registered");
        // 4 sliders (gain, centre x, centre y, invert) + 3 rgb lanes.
        assert_eq!(def.params.len(), 7, "{:?}", def.params);
        assert!(def.params.iter().any(|p| p.label == "Gain"));
        assert!(def.params.iter().any(|p| p.label == "Centre x"));
        assert!(def.params.iter().any(|p| p.label == "Tint.r"));

        // An instance's knob reaches the shader through expansion.
        let mut g = EditorGraph::default();
        let plate = g.add(NodeKind::Transform(Default::default()), egui::Pos2::ZERO);
        let inst = g.add(
            NodeKind::Custom { type_id: "t.shaderui".into(), params: Default::default() },
            egui::Pos2::ZERO,
        );
        g.edges.push((inst, 0, plate));
        g.view = Some(inst);
        let gain_knob = def
            .params
            .iter()
            .find(|p| p.label == "Gain")
            .expect("gain knob")
            .name
            .clone();
        if let NodeKind::Custom { params, .. } = &mut g.nodes.get_mut(&inst).unwrap().kind {
            params.insert(gain_knob, 2.75);
        }
        let flat = tools::expand(&g, None).expect("expands");
        let view = flat.view.unwrap();
        let gain = crate::anim::Channel::Shader(crate::anim::intern("gain"));
        assert_eq!(crate::anim::get(&flat.nodes[&view].kind, gain), Some(2.75));
        // ...and the shader itself came along.
        assert!(matches!(&flat.nodes[&view].kind, NodeKind::Shader(p) if p.source == MIXED));
    }

    #[test]
    fn applying_keeps_values_by_name() {
        let mut p = ShaderParams::default();
        p.values.insert("amount".into(), [3.0, 0.0, 0.0, 0.0]);
        p.set_source(EX_RADIAL.to_string());
        // `amount` survives (both shaders declare it); `tint` is gone and
        // `centre` arrives at its declared default.
        assert_eq!(p.values["amount"], [3.0, 0.0, 0.0, 0.0]);
        assert!(!p.values.contains_key("tint"));
        assert_eq!(p.values["centre"], [0.5, 0.5, 0.0, 0.0]);
    }
}
