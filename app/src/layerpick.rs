//! Ctrl-click the picture to grab the layer under the pointer, the way
//! Photoshop's auto-select does: the viewer works out which BRANCH of the
//! comp put that pixel there — by stacking order and by alpha, so clicking
//! through a cutout's transparent corner lands on what is behind it — and
//! selects that branch's Transform, which brings up its manipulator and
//! raises its panel.
//!
//! Two halves:
//! - [`candidates`] is pure graph: walk down from the view target, and at
//!   every Blend / Layer list the inputs top of the stack first (A before B,
//!   layer 1 before layer 2), recursing into nested merges. Each candidate
//!   carries the map from its own pixels to the viewer's, so the click can
//!   be carried back into the branch's space and a branch that does not
//!   even cover the point is rejected on the CPU.
//! - [`Gfx::run_pending_pick`] asks the GPU only about the survivors: it
//!   renders the comp retargeted at each candidate and reads back ONE
//!   texel's alpha, top first, stopping at the first that is there. It runs
//!   on the click and never per frame (manifesto 2); a retarget is a
//!   pipeline-cache hit after the first pick of a comp, a ~100ms compile
//!   before that.

use crate::editor::EditorGraph;
use crate::{H3, IDENTITY_H, NodeKind, h_mul};

/// Alpha at or below this is a hole you click THROUGH.
const ALPHA_HIT: f32 = 0.02;

/// One branch that could own the clicked pixel.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// The node wired into the merge: its output alpha decides whether the
    /// branch covers the pixel.
    pub probe: u64,
    /// `probe`'s output pixels -> the viewer's output pixels.
    pub to_view: H3,
    /// View target down to `probe`, both ends included.
    pub path: Vec<u64>,
    /// The stack and layer index, when the merge is a Layer node.
    pub layer: Option<(u64, usize)>,
}

/// What a viewer pick brings up besides the selection, read off the
/// click's modifiers: ctrl shows the node's HUD, cmd raises its properties,
/// both do both. A pick SOLOS what it brings up — that node's controls and
/// nobody else's — and shift ADDS instead, to the selection and to what is
/// already up. Shared by every pick in the viewer — the picture, a
/// manipulator's pivot — so one chord means one thing. (cmd is the Mac key;
/// elsewhere ctrl is the only chord and shows the HUD.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reveal {
    pub hud: bool,
    pub props: bool,
    pub add: bool,
}

impl Reveal {
    pub fn from_mods(m: egui::Modifiers) -> Option<Self> {
        let r = Self { hud: m.ctrl, props: m.mac_cmd, add: m.shift };
        (r.hud || r.props).then_some(r)
    }

    /// This frame's click, if it was a pick. macOS turns a ctrl-click into
    /// a secondary click on the way through, so both buttons count.
    pub fn clicked(ui: &egui::Ui) -> Option<Self> {
        ui.input(|i| {
            (i.pointer.primary_clicked() || i.pointer.secondary_clicked())
                .then(|| Self::from_mods(i.modifiers))
                .flatten()
        })
    }
}

/// What a pick selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Node(u64),
    Layer(u64, usize),
}

/// The image input a single-input walk follows: the first wired input that
/// is not a matte.
fn image_input(g: &EditorGraph, id: u64) -> Option<(usize, u64)> {
    let n = g.node(id)?;
    if n.disabled {
        // A bypassed node shows what it passes through (a Blend: its bg).
        let i = n.kind.bypass_input();
        return g.input_of(id, i).map(|s| (i, s));
    }
    (0..n.kind.input_count())
        .filter(|i| !n.kind.is_side_input(*i))
        .find_map(|i| g.input_of(id, i).map(|s| (i, s)))
}

/// Carry a probe up through the Transforms and CornerPins that place it:
/// they move pixels OUT of their own frame (a reformat Crop's box moved
/// across a Layer canvas lands far outside the box-sized format it keeps),
/// so the frame test and the alpha read happen where the pixels come FROM.
fn lift(g: &EditorGraph, mut probe: u64, mut acc: H3, step: &dyn Fn(u64, usize) -> H3) -> (u64, H3) {
    for _ in 0..64 {
        let Some(n) = g.node(probe) else { break };
        if n.disabled || !matches!(n.kind, NodeKind::Transform(_) | NodeKind::CornerPin(_)) {
            break;
        }
        let Some((i, src)) = image_input(g, probe) else { break };
        acc = h_mul(&acc, &step(probe, i));
        probe = g.resolve_dot(src).unwrap_or(src);
    }
    (probe, acc)
}

/// Every branch that can own a pixel of `view`, top of the stack first.
/// `step(node, input)` is what `node` does to the pixels arriving on
/// `input` (input px -> node output px).
pub fn candidates(
    g: &EditorGraph,
    view: u64,
    step: &dyn Fn(u64, usize) -> H3,
) -> Vec<Candidate> {
    #[allow(clippy::too_many_arguments)]
    fn walk(
        g: &EditorGraph,
        id: u64,
        acc: H3,
        path: &mut Vec<u64>,
        layer: Option<(u64, usize)>,
        step: &dyn Fn(u64, usize) -> H3,
        out: &mut Vec<Candidate>,
        seen: &mut std::collections::HashSet<u64>,
    ) -> bool {
        if !seen.insert(id) {
            return false;
        }
        let Some(n) = g.node(id) else { return false };
        path.push(id);
        let inputs: Option<Vec<usize>> = match &n.kind {
            _ if n.disabled => None,
            // A over B: A is on top.
            NodeKind::Blend { .. } => Some(vec![0, 1]),
            // Port order IS stack order, 0 = top.
            NodeKind::Layer(_) => Some((0..n.kind.input_count()).collect()),
            _ => None,
        };
        let found = match inputs {
            Some(inputs) => {
                let is_layer = matches!(n.kind, NodeKind::Layer(_));
                for i in inputs {
                    let Some(src) = g.input_of(id, i) else { continue };
                    let acc = h_mul(&acc, &step(id, i));
                    let layer = if is_layer { Some((id, i)) } else { layer };
                    let before = out.len();
                    walk(g, src, acc, path, layer, step, out, seen);
                    // No merge further down: this input IS a layer.
                    if out.len() == before {
                        let (probe, to_view) =
                            lift(g, g.resolve_dot(src).unwrap_or(src), acc, step);
                        let mut p = path.clone();
                        p.push(src);
                        out.push(Candidate { probe, to_view, path: p, layer });
                    }
                }
                true
            }
            None => match image_input(g, id) {
                Some((i, src)) => {
                    walk(g, src, h_mul(&acc, &step(id, i)), path, layer, step, out, seen)
                }
                None => false,
            },
        };
        path.pop();
        found
    }
    let mut out = Vec::new();
    let mut seen = Default::default();
    let found = walk(g, view, IDENTITY_H, &mut Vec::new(), None, step, &mut out, &mut seen);
    if !found {
        // No merge anywhere: the whole picture is one layer.
        out.push(Candidate { probe: view, to_view: IDENTITY_H, path: vec![view], layer: None });
    }
    out
}

/// What to select for a branch that owns the pixel: the first Transform
/// walking UP the branch from where it enters its merge (a Layer stack's
/// layers included — placement is always an upstream node's); failing that
/// the layer itself, so the stack's panel says which one it was; failing
/// that the nearest Transform back down toward the viewer (one moving a
/// whole nested comp); failing that the branch's own node, so the pick
/// still says which branch it was.
pub fn owner(g: &EditorGraph, c: &Candidate) -> Pick {
    let is_xf = |id: u64| matches!(g.node(id).map(|n| &n.kind), Some(NodeKind::Transform(_)));
    let mut cur = c.path.last().copied();
    let mut seen = std::collections::HashSet::new();
    while let Some(id) = cur {
        if !seen.insert(id) {
            break;
        }
        if is_xf(id) {
            return Pick::Node(id);
        }
        cur = image_input(g, id).map(|(_, s)| s);
    }
    if let Some((stack, i)) = c.layer {
        return Pick::Layer(stack, i);
    }
    if let Some(id) = c.path.iter().rev().skip(1).copied().find(|id| is_xf(*id)) {
        return Pick::Node(id);
    }
    Pick::Node(c.probe)
}

impl crate::Gfx {
    /// Resolve a ctrl-click queued by the viewer (`pending_pick`, in the
    /// viewer's output pixels). Runs between frames, before the frame's own
    /// graph is built: the probes borrow the compositor's output texture,
    /// and the frame that follows renders over them.
    pub(crate) fn run_pending_pick(&mut self) {
        let Some((px, reveal)) = self.pending_pick.take() else { return };
        let dual = self.dual_view;
        let Some(view) = self.egraph.live_view(dual).and_then(|v| self.egraph.resolve_dot(v))
        else {
            return;
        };
        let t = self.eval_tick();
        let (ow, oh) = self.comp.output_size().unwrap_or((1920, 1080));
        let res = |id: u64| self.node_res_f32(id).unwrap_or((ow as f32, oh as f32));
        let step = |id: u64, _input: usize| crate::node_step_h(&self.egraph, id, &res, t);
        let cands = candidates(&self.egraph, view, &step);
        let before = self.comp.output_size();
        let mut hit = None;
        for c in &cands {
            // CPU first: carry the click into the branch's own pixels and
            // throw out every branch whose frame is not even under it.
            let Some((w, h)) = self
                .node_resolution(c.probe, &mut Default::default())
                .map(|(w, h)| (w.clamp(16, 16384), h.clamp(16, 16384)))
            else {
                continue;
            };
            let p = crate::apply_h(&engine::codegen::mat3_inv(&c.to_view), px);
            if !(p.x >= 0.0 && p.y >= 0.0 && p.x < w as f32 && p.y < h as f32) {
                continue;
            }
            // Then the GPU, for the alpha. A branch that cannot be probed
            // (it did not compile, or lowering folded it away) goes by its
            // frame alone rather than being unclickable.
            match self.probe_alpha(c.probe, (p.x as u32, p.y as u32), (w, h)) {
                Some(a) if a <= ALPHA_HIT => continue,
                _ => {
                    hit = Some(c);
                    break;
                }
            }
        }
        // Put the viewer's own output back; the frame that follows re-renders
        // it (and a render-cached display re-shows its frame).
        if let Some((w, h)) = before {
            if self.comp.output_size() != Some((w, h)) {
                self.comp.ensure_output(w, h);
                self.blit.invalidate();
            }
        }
        self.last_engine_graph = None;
        if self.render_play.is_some() {
            self.refresh_display = true;
        }
        let Some(c) = hit else { return };
        match owner(&self.egraph, c) {
            Pick::Node(id) => self.reveal_node(id, reveal),
            Pick::Layer(stack, i) => {
                self.layer_sel.insert(stack, i);
                self.reveal_node(stack, reveal);
            }
        }
    }

    /// Render the comp with `node` as the view target and read the alpha
    /// of one texel. Blocking; for a click only.
    fn probe_alpha(&mut self, node: u64, px: (u32, u32), size: (u32, u32)) -> Option<f32> {
        let tick = self.eval_tick();
        let mut g = self.egraph.to_engine_graph_at(Some(tick));
        if !g.nodes.contains_key(&node) {
            return None;
        }
        g.view = Some(node);
        g.view_out = 0;
        self.stamp_formats(&mut g);
        self.apply_bake_state(&mut g);
        if self.comp.output_size() != Some(size) {
            self.comp.ensure_output(size.0, size.1);
        }
        if !self.comp.set_graph(&g) {
            return None;
        }
        self.comp.set_clock(self.shown_tick as f32, self.fps);
        self.comp.set_paint_frame(tick);
        self.comp.write_params(&g);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("layer-pick") });
        self.comp.encode_comp(&mut enc, &self.timer);
        // Stage chains come up with zeroed uniforms (see `params_stale`).
        if self.comp.params_stale() {
            self.comp.write_params(&g);
        }
        self.ctx.queue.submit([enc.finish()]);
        self.probe_output_box(px.0.min(size.0 - 1), px.1.min(size.1 - 1), 1, 1, 3)
    }
}

impl crate::Gfx {
    /// --test-layerpick: a see-through grid over an opaque one, each behind
    /// its own Transform. Ctrl-clicking a LINE of the top grid must select
    /// its Transform; clicking BETWEEN its lines — inside its frame, but
    /// where its alpha is zero — must click through to the bottom one. That
    /// second click is the one a frame-only test would get wrong. Each chord
    /// is checked for what it brings up: ctrl the HUD alone, cmd the
    /// properties, both both.
    pub(crate) fn drive_test_layerpick(&mut self, raw: &mut egui::RawInput) {
        use engine::graph::{BlendOp, GridParams, TransformParams};
        let Some((stage, mut frames, [xf_fg, xf_bg])) = self.test.layerpick else { return };
        frames += 1;
        if frames > 900 {
            println!("LAYERPICK TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let ctrl = egui::Modifiers { ctrl: true, ..Default::default() };
        let cmd = egui::Modifiers { mac_cmd: true, command: true, ..Default::default() };
        let both = egui::Modifiers { ctrl: true, mac_cmd: true, command: true, ..Default::default() };
        let cmd_add = egui::Modifiers { mac_cmd: true, command: true, shift: true, ..Default::default() };
        // A point of the COMP (the viewer's output pixels) on screen, taken
        // at the press and reused for the release: a panel that moves in
        // between would otherwise turn the click into a drag.
        let screen = |s: &Self, at: egui::Vec2| {
            let (ow, oh) = s.comp.output_size().unwrap_or((1920, 1080));
            let r = s.image_rect;
            r.min + egui::vec2(at.x / ow as f32 * r.width(), at.y / oh as f32 * r.height())
        };
        let click = |raw: &mut egui::RawInput, pos: egui::Pos2, m: egui::Modifiers, down: bool| {
            raw.events.insert(0, egui::Event::ModifiersChanged(m));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: m,
            });
        };
        // What the last pick left: (selected, properties raised for it).
        let state = |s: &Self| {
            let sel = s.selected.first().copied();
            (sel, sel.is_some() && s.props_visible && s.prop_order.first() == sel.as_ref())
        };
        let check = |s: &Self, what: &str, want: u64, props: bool| {
            let (sel, raised) = state(s);
            if sel != Some(want) || raised != props {
                println!(
                    "LAYERPICK TEST FAIL: {what} selected {sel:?} (properties raised {raised}), \
                     wanted {want} (properties raised {props})"
                );
                std::process::exit(1);
            }
        };
        // Lines every 400px, 40px thick, centred on the multiples.
        let on_line = egui::vec2(800.0, 600.0);
        let between = egui::vec2(600.0, 600.0);
        // (point, chord) per click, after stage 0.
        let clicks = [(on_line, ctrl), (between, cmd), (on_line, both), (between, cmd_add)];
        let mut next = stage;
        match stage {
            0 if frames > 10 && self.layout_settled() => {
                let at = |x: f32| egui::pos2(x, 400.0);
                let fg = self.egraph.add(
                    NodeKind::Grid(GridParams {
                        spacing: [400.0, 400.0],
                        width: 40.0,
                        ..Default::default()
                    }),
                    at(0.0),
                );
                let bg = self.egraph.add(
                    NodeKind::Grid(GridParams { back: [0.2, 0.2, 0.2, 1.0], ..Default::default() }),
                    at(200.0),
                );
                let xf_fg = self.egraph.add(NodeKind::Transform(TransformParams::default()), at(0.0));
                let xf_bg = self.egraph.add(NodeKind::Transform(TransformParams::default()), at(200.0));
                let top = self.egraph.add(
                    NodeKind::Blend { op: BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] },
                    at(100.0),
                );
                self.egraph.connect(xf_fg, 0, fg);
                self.egraph.connect(xf_bg, 0, bg);
                self.egraph.connect(top, 0, xf_fg);
                self.egraph.connect(top, 1, xf_bg);
                self.egraph.view = Some(top);
                self.selected = Vec::new();
                self.prop_order.clear();
                self.props_visible = false;
                self.test.layerpick = Some((1, 0, [xf_fg, xf_bg]));
                return;
            }
            // Odd stages press, even stages release, and the stage after a
            // release checks it and presses the next.
            1 | 3 | 5 | 7 | 9 if frames > if stage == 1 { 30 } else { 6 } && self.layout_settled() => {
                let k = (stage as usize - 1) / 2;
                if k > 0 {
                    if k < 4 {
                        let (what, want, props) = [
                            ("ctrl-click on the top grid's line", xf_fg, false),
                            ("cmd-click in a HOLE of the top grid (alpha ignored?)", xf_bg, true),
                            ("ctrl+cmd-click on the top grid's line", xf_fg, true),
                        ][k - 1];
                        check(self, what, want, props);
                        // Solo: the rack holds the clicked node alone.
                        if props && self.prop_order != vec![want] {
                            println!(
                                "LAYERPICK TEST FAIL: {what} did not solo the properties \
                                 (rack {:?})",
                                self.prop_order
                            );
                            std::process::exit(1);
                        }
                    } else {
                        // shift+cmd ADDS: both selected, both panels up,
                        // the new one in front.
                        let both_sel = self.selected.contains(&xf_fg) && self.selected.contains(&xf_bg);
                        if !both_sel || self.prop_order != vec![xf_bg, xf_fg] {
                            println!(
                                "LAYERPICK TEST FAIL: shift+cmd-click did not add \
                                 (selected {:?}, rack {:?})",
                                self.selected, self.prop_order
                            );
                            std::process::exit(1);
                        }
                    }
                }
                match clicks.get(k) {
                    Some(&(at, m)) => {
                        self.layer_grab = screen(self, at);
                        click(raw, self.layer_grab, m, true);
                        next = stage + 1;
                    }
                    None => {
                        println!(
                            "LAYERPICK TEST PASS: ctrl-click on the top grid's line picked its \
                             Transform (HUD, no properties); cmd-click in a gap clicked through \
                             to the background's and raised its properties; ctrl+cmd did both; \
                             each soloed, and shift+cmd added instead"
                        );
                        std::process::exit(0);
                    }
                }
            }
            2 | 4 | 6 | 8 if frames > 2 => {
                let (_, m) = clicks[(stage as usize - 2) / 2];
                click(raw, self.layer_grab, m, false);
                next = stage + 1;
            }
            _ => {}
        }
        if next != stage {
            frames = 0;
        }
        self.test.layerpick = Some((next, frames, [xf_fg, xf_bg]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::{BlendOp, TransformParams};

    fn blend() -> NodeKind {
        NodeKind::Blend { op: BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] }
    }

    /// A source with no inputs (a plate, as far as the walk cares).
    fn read() -> NodeKind {
        NodeKind::Grid(Default::default())
    }


    /// Two plates over a background, one of them nested:
    /// ```text
    ///   fg -> xf_fg ---\
    ///                   blend_in -> xf_all --\
    ///   mid ----------/                        blend_top (viewed)
    ///   bg ----------------------------------/
    /// ```
    fn comp() -> (EditorGraph, [u64; 8]) {
        let mut g = EditorGraph::default();
        let z = egui::Pos2::ZERO;
        let fg = g.add(read(), z);
        let xf_fg = g.add(NodeKind::Transform(TransformParams::default()), z);
        let mid = g.add(read(), z);
        let blend_in = g.add(blend(), z);
        let xf_all = g.add(NodeKind::Transform(TransformParams::default()), z);
        let bg = g.add(read(), z);
        let top = g.add(blend(), z);
        g.connect(xf_fg, 0, fg);
        g.connect(blend_in, 0, xf_fg);
        g.connect(blend_in, 1, mid);
        g.connect(xf_all, 0, blend_in);
        g.connect(top, 0, xf_all);
        g.connect(top, 1, bg);
        g.view = Some(top);
        (g, [fg, xf_fg, mid, blend_in, xf_all, bg, top, 0])
    }

    #[test]
    fn candidates_run_top_of_the_stack_first() {
        let (g, [fg, _, mid, _, _, bg, top, _]) = comp();
        let c = candidates(&g, top, &|_, _| IDENTITY_H);
        let probes: Vec<u64> = c.iter().map(|c| c.probe).collect();
        // fg is probed ABOVE its Transform (`lift`), where its pixels come from.
        assert_eq!(probes, vec![fg, mid, bg], "A before B, nested merges first");
    }

    #[test]
    fn a_pick_selects_the_branch_transform_then_the_enclosing_one() {
        let (g, [_, xf_fg, _, _, xf_all, bg, top, _]) = comp();
        let c = candidates(&g, top, &|_, _| IDENTITY_H);
        // fg has its own Transform.
        assert_eq!(owner(&g, &c[0]), Pick::Node(xf_fg));
        // mid has none of its own: the one moving the nested comp.
        assert_eq!(owner(&g, &c[1]), Pick::Node(xf_all));
        // bg has no Transform anywhere: the branch itself.
        assert_eq!(owner(&g, &c[2]), Pick::Node(bg));
    }

    #[test]
    fn the_click_is_carried_through_every_transform_on_the_way_down() {
        let (g, [_, xf_fg, _, _, xf_all, _, top, _]) = comp();
        let shift = |x| [[1.0, 0.0, x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let step = |id: u64, _: usize| {
            if id == xf_fg {
                shift(5.0)
            } else if id == xf_all {
                shift(10.0)
            } else {
                IDENTITY_H
            }
        };
        let c = candidates(&g, top, &step);
        // The fg branch is probed above its own Transform, so both map its
        // pixels; mid shares only the outer one; bg sits under neither.
        assert_eq!(c[0].to_view[0][2], 15.0);
        assert_eq!(c[1].to_view[0][2], 10.0);
        assert_eq!(c[2].to_view[0][2], 0.0);
    }

    #[test]
    fn a_layer_is_moved_by_its_upstream_transform() {
        let mut g = EditorGraph::default();
        let z = egui::Pos2::ZERO;
        let (a, b) = (g.add(read(), z), g.add(read(), z));
        let xf = g.add(NodeKind::Transform(TransformParams::default()), z);
        g.connect(xf, 0, a);
        let stack = g.add(NodeKind::layer(), z);
        g.connect(stack, 0, xf);
        g.connect(stack, 1, b);
        let c = candidates(&g, stack, &|_, _| IDENTITY_H);
        assert_eq!(c.iter().map(|c| c.probe).collect::<Vec<_>>(), vec![a, b]);
        // The top layer's Transform is what moves it, so that is the pick;
        // a layer with none upstream picks the layer, for its panel.
        assert_eq!(owner(&g, &c[0]), Pick::Node(xf));
        assert_eq!(owner(&g, &c[1]), Pick::Layer(stack, 1));
    }

    /// A Transform moves a reformat Crop's box far outside the box-sized
    /// frame it keeps: the click is tested where the pixels come from, the
    /// Crop, and the pick is still the Transform.
    #[test]
    fn a_crop_moved_out_of_its_own_frame_is_still_picked() {
        let mut g = EditorGraph::default();
        let z = egui::Pos2::ZERO;
        let (a, b) = (g.add(read(), z), g.add(read(), z));
        let crop = g.add(NodeKind::SoftCrop(Default::default()), z);
        let xf = g.add(NodeKind::Transform(TransformParams::default()), z);
        g.connect(crop, 0, a);
        g.connect(xf, 0, crop);
        let stack = g.add(NodeKind::layer(), z);
        g.connect(stack, 0, xf);
        g.connect(stack, 1, b);
        let step = |id: u64, _: usize| {
            if id == xf { [[1.0, 0.0, 2000.0], [0.0, 1.0, 900.0], [0.0, 0.0, 1.0]] } else { IDENTITY_H }
        };
        let c = candidates(&g, stack, &step);
        assert_eq!(c[0].probe, crop);
        assert_eq!((c[0].to_view[0][2], c[0].to_view[1][2]), (2000.0, 900.0));
        assert_eq!(owner(&g, &c[0]), Pick::Node(xf));
    }

    #[test]
    fn a_comp_with_no_merge_is_one_layer() {
        let mut g = EditorGraph::default();
        let r = g.add(read(), egui::Pos2::ZERO);
        let xf = g.add(NodeKind::Transform(TransformParams::default()), egui::Pos2::ZERO);
        g.connect(xf, 0, r);
        let c = candidates(&g, xf, &|_, _| IDENTITY_H);
        assert_eq!(c.len(), 1);
        assert_eq!(owner(&g, &c[0]), Pick::Node(xf));
    }
}
