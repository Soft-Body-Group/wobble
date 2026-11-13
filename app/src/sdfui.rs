//! The SDF node's app side: its glyph, its panels (the node in the comp
//! and every node inside its graph), and the round trip between an
//! `SdfGraph` and the editor graph a dive swaps in. The engine half —
//! the graph, the node set and the WGSL — is `engine/src/sdf.rs`.

use crate::editor::{self, EditorGraph};
use crate::{anim, colorpicker, widgets};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};
use engine::graph::NodeKind;
use engine::sdf::{CombineOp, MergeOp, SdfGraph, SdfNode, Side};

/// A filled body, a dot, and two iso-lines round it: a distance, drawn.
/// `colors/src/glyphs.rs` keeps a copy (`glyph_parity` checks it).
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        Rect::from_center_size(c, Vec2::splat(r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    painter.circle(c, r * 0.2, stroke.color, Stroke::NONE);
    painter.circle_stroke(c, r * 0.42, stroke);
    painter.circle_stroke(c, r * 0.64, stroke);
}

/// An SDF graph as the node editor holds it: every node a
/// `NodeKind::SdfNode`, so selection, wiring, undo and the tile layout work
/// on it unchanged — the noise field's arrangement.
pub fn to_editor(f: &SdfGraph) -> EditorGraph {
    let mut g = EditorGraph::default();
    for (i, (id, node)) in f.nodes.iter().enumerate() {
        let p = f.layout.get(id).copied().unwrap_or([120.0 * i as f32, 60.0 * i as f32]);
        g.nodes.insert(*id, editor::ENode::new(NodeKind::SdfNode(node.clone()), egui::pos2(p[0], p[1])));
    }
    for ((dst, port), src) in &f.edges {
        g.edges.push((*dst, *port, *src));
    }
    g.view = f.out;
    g.next_id = f.next_id.max(f.nodes.keys().copied().max().unwrap_or(0) + 1);
    g
}

/// The other direction: anything that is not an SDF node is dropped.
pub fn from_editor(g: &EditorGraph) -> SdfGraph {
    let mut f = SdfGraph::default();
    for (id, n) in &g.nodes {
        if let NodeKind::SdfNode(k) = &n.kind {
            f.nodes.insert(*id, k.clone());
            f.layout.insert(*id, [n.pos.x, n.pos.y]);
        }
    }
    for (dst, port, src) in &g.edges {
        if f.nodes.contains_key(dst) && f.nodes.contains_key(src) {
            f.edges.insert((*dst, *port), *src);
        }
    }
    f.next_id = g.next_id;
    f
}

/// A fresh graph for a node opened empty: a circle, filled, into the
/// Output — something on screen straight away, and the shape of the
/// paradigm (shape -> effect -> output) visible at a glance.
pub fn starter() -> SdfGraph {
    let mut g = SdfGraph::default();
    g.nodes.insert(1, SdfNode::Spline {
        points: Vec::new(),
        mode: engine::graph::SplineMode::CatmullRom,
        closed: true,
        offset: [0.0, 0.0],
    });
    g.nodes.insert(2, SdfNode::Fill { color: [1.0, 1.0, 1.0, 1.0], soft: 0.0 });
    g.nodes.insert(3, SdfNode::Output);
    g.edges.insert((2, 0), 1);
    g.edges.insert((3, 0), 2);
    g.layout.insert(1, [0.0, 0.0]);
    g.layout.insert(2, [0.0, 110.0]);
    g.layout.insert(3, [0.0, 220.0]);
    g.next_id = 4;
    g
}

/// The SDF node's own panel: its format, and every Slider inside as a row
/// of its own (a `Custom` channel, so it keys, cables and drags out like
/// any other).
pub fn owner_panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    node: u64,
    p: &mut engine::graph::SdfParams,
    input_wired: bool,
) -> Option<widgets::HudSlider> {
    let mut hud = None;
    if input_wired {
        ui.weak("drawing over its input, at the input's format");
    } else {
        widgets::gen_format_row(ui, ("sdf-format", node), &mut p.size, &mut p.inherit);
    }
    let exposed = p.graph.exposed();
    if !exposed.is_empty() {
        ui.separator();
    }
    for (name, is_input) in &exposed {
        if *is_input {
            continue;
        }
        let ch = anim::Channel::Custom(anim::intern(name));
        let mut val = p.graph.exposed_value(name).unwrap_or(0.0);
        let before = val;
        if let Some(h) = widgets::slider(ui, rcx, name, &mut val, -100.0, 100.0, &[ch]) {
            hud = Some(h);
        }
        if val != before {
            p.graph.set_exposed(name, val);
        }
    }
    let inputs: Vec<&str> =
        exposed.iter().filter(|(_, i)| *i).map(|(nm, _)| nm.as_str()).collect();
    if !inputs.is_empty() {
        ui.weak(format!("data inputs: {} — cable them in link mode (E)", inputs.join(", ")));
    }
    ui.weak(if p.graph.nodes.is_empty() {
        "empty — open it (↩) to draw a shape"
    } else {
        "open it (↩) to edit the shapes and effects"
    });
    hud
}

/// One node INSIDE an SDF graph. Its sockets are `Channel::Field(i)` rows
/// (key dot, drag-out, patch jack); a socket wired to another node is
/// greyed, because the wire replaces the number per pixel.
#[allow(clippy::too_many_arguments)]
pub fn node_panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    node: u64,
    n: &mut SdfNode,
    wired: &[bool],
    picked: &[usize],
    spec: colorpicker::PickSpec,
) -> Option<widgets::HudSlider> {
    let mut hud = None;
    let mut row = |ui: &mut egui::Ui, rcx: &mut widgets::RowCtx, label: &str, v: &mut f32, lo: f32, hi: f32, i: usize| {
        if wired.get(i).copied().unwrap_or(false) {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [76.0, 16.0],
                    egui::Label::new(egui::RichText::new(label).small().color(crate::theme::cable())),
                );
                ui.add_enabled(false, egui::DragValue::new(v).update_while_editing(false));
                ui.weak("driven by a wire");
            });
            return;
        }
        if let Some(h) = widgets::slider(ui, rcx, label, v, lo, hi, &[anim::Channel::Field(i as u16)]) {
            hud = Some(h);
        }
    };
    let colour = |ui: &mut egui::Ui, label: &str, c: &mut [f32; 4]| {
        ui.horizontal(|ui| {
            ui.add_sized([76.0, 16.0], egui::Label::new(egui::RichText::new(label).small()));
            colorpicker::button(ui, (node, "sdf", label), c, spec);
        });
    };
    match n {
        SdfNode::Spline { points, mode, closed, offset } => {
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(("sdf-spline-mode", node))
                    .width(96.0)
                    .selected_text(mode.name())
                    .show_ui(ui, |ui| {
                        for m in engine::graph::SplineMode::ALL {
                            ui.selectable_value(mode, m, m.name());
                        }
                    });
                ui.checkbox(closed, "closed")
                    .on_hover_text("closed: a shape with an inside; open: a path, distance to the curve");
            });
            row(ui, rcx, "offset x", &mut offset[0], -500.0, 500.0, 0);
            row(ui, rcx, "offset y", &mut offset[1], -500.0, 500.0, 1);
            crate::tension_row(ui, points, *mode, picked);
            ui.weak(if points.is_empty() {
                "click in the viewer to lay points down"
            } else {
                "drag points in the viewer; S-click adds, D-click removes"
            });
        }
        SdfNode::Circle { center, radius } => {
            row(ui, rcx, "centre x", &mut center[0], 0.0, 1920.0, 0);
            row(ui, rcx, "centre y", &mut center[1], 0.0, 1080.0, 1);
            row(ui, rcx, "radius", radius, 0.0, 800.0, 2);
        }
        SdfNode::Rect { center, size, round } => {
            row(ui, rcx, "centre x", &mut center[0], 0.0, 1920.0, 0);
            row(ui, rcx, "centre y", &mut center[1], 0.0, 1080.0, 1);
            row(ui, rcx, "width", &mut size[0], 0.0, 1920.0, 2);
            row(ui, rcx, "height", &mut size[1], 0.0, 1080.0, 3);
            row(ui, rcx, "round", round, 0.0, 300.0, 4);
        }
        SdfNode::Combine { op, smooth } => {
            ui.horizontal(|ui| {
                for o in CombineOp::ALL {
                    ui.selectable_value(op, o, o.name());
                }
            });
            row(ui, rcx, "smooth", smooth, 0.0, 200.0, 2);
            ui.weak("smooth melts the seam over that many pixels");
        }
        SdfNode::Offset { amount } => {
            row(ui, rcx, "amount", amount, -200.0, 200.0, 1);
            ui.weak("grow (+) or shrink (−) the shape");
        }
        SdfNode::Shell { width } => row(ui, rcx, "width", width, 0.0, 200.0, 1),
        SdfNode::Morph { t } => row(ui, rcx, "t", t, 0.0, 1.0, 2),
        SdfNode::Transform { translate, rotate, scale, pivot } => {
            row(ui, rcx, "move x", &mut translate[0], -1000.0, 1000.0, 1);
            row(ui, rcx, "move y", &mut translate[1], -1000.0, 1000.0, 2);
            row(ui, rcx, "rotate", rotate, -180.0, 180.0, 3);
            row(ui, rcx, "scale", scale, 0.05, 4.0, 4);
            widgets::plain_slider(ui, "pivot x", &mut pivot[0], 0.0, 1920.0, false);
            widgets::plain_slider(ui, "pivot y", &mut pivot[1], 0.0, 1080.0, false);
        }
        SdfNode::Displace { amount, size, seed, speed, octaves } => {
            row(ui, rcx, "amount", amount, 0.0, 100.0, 1);
            row(ui, rcx, "size", size, 1.0, 600.0, 2);
            widgets::plain_slider(ui, "seed", seed, 0.0, 256.0, true);
            widgets::plain_slider(ui, "speed", speed, 0.0, 0.5, false)
                .on_hover_text("how fast the wobble evolves, per frame");
            let mut o = *octaves as f32;
            if widgets::plain_slider(ui, "octaves", &mut o, 1.0, engine::field::FBM_MAX_OCTAVES as f32, true)
                .changed()
            {
                *octaves = o.round() as u32;
            }
        }
        SdfNode::Const { value, name } | SdfNode::Input { name, value } => {
            ui.horizontal(|ui| {
                ui.add_sized([76.0, 16.0], egui::Label::new(egui::RichText::new("name").small()));
                ui.text_edit_singleline(name);
            });
            if let Some(h) =
                widgets::slider(ui, rcx, "value", value, -100.0, 100.0, &[anim::Channel::Field(0)])
            {
                hud = Some(h);
            }
            ui.weak("wire it into any socket; it shows on the SDF node by name");
        }
        SdfNode::Fill { color, soft } => {
            colour(ui, "colour", color);
            row(ui, rcx, "softness", soft, 0.0, 100.0, 1);
        }
        SdfNode::Stroke { color, width, soft } => {
            colour(ui, "colour", color);
            row(ui, rcx, "width", width, 0.0, 100.0, 1);
            row(ui, rcx, "softness", soft, 0.0, 100.0, 2);
        }
        SdfNode::Glow { color, radius, intensity } => {
            colour(ui, "colour", color);
            row(ui, rcx, "radius", radius, 0.0, 400.0, 1);
            row(ui, rcx, "intensity", intensity, 0.0, 8.0, 2);
            ui.weak("light, not paint: it adds to what it lands on");
        }
        SdfNode::Shadow { color, offset, soft } => {
            colour(ui, "colour", color);
            row(ui, rcx, "offset x", &mut offset[0], -200.0, 200.0, 1);
            row(ui, rcx, "offset y", &mut offset[1], -200.0, 200.0, 2);
            row(ui, rcx, "softness", soft, 0.0, 200.0, 3);
        }
        SdfNode::Ripples { color, spacing, width, speed, falloff, side } => {
            colour(ui, "colour", color);
            ui.horizontal(|ui| {
                for s in Side::ALL {
                    ui.selectable_value(side, s, s.name());
                }
            });
            row(ui, rcx, "spacing", spacing, 2.0, 200.0, 1);
            row(ui, rcx, "width", width, 0.0, 40.0, 2);
            row(ui, rcx, "speed", speed, -10.0, 10.0, 3);
            row(ui, rcx, "falloff", falloff, 0.0, 800.0, 4);
            ui.weak("speed in pixels a frame; falloff 0 never fades");
        }
        SdfNode::Bevel { color, width, angle, elevation, depth } => {
            colour(ui, "colour", color);
            row(ui, rcx, "width", width, 0.0, 200.0, 1);
            row(ui, rcx, "angle", angle, -180.0, 180.0, 2);
            row(ui, rcx, "elevation", elevation, 0.0, 90.0, 3);
            row(ui, rcx, "depth", depth, 0.0, 4.0, 4);
        }
        SdfNode::Refract { amount, width } => {
            row(ui, rcx, "amount", amount, -100.0, 100.0, 1);
            row(ui, rcx, "width", width, 0.0, 300.0, 2);
            ui.weak("the SDF node's input, bent at the rim — it is fused a second time there");
        }
        SdfNode::Ramp { inner, outer, range } => {
            colour(ui, "inner", inner);
            colour(ui, "outer", outer);
            row(ui, rcx, "range", range, 1.0, 800.0, 1);
        }
        SdfNode::Merge { op, mix } => {
            ui.horizontal(|ui| {
                for o in MergeOp::ALL {
                    ui.selectable_value(op, o, o.name());
                }
            });
            row(ui, rcx, "mix", mix, 0.0, 1.0, 2);
        }
        SdfNode::ImageIn => {
            ui.weak("the SDF node's input, at this pixel");
        }
        SdfNode::Output => {
            ui.weak("what the SDF node draws, laid over its input");
        }
    }
    hud
}

/// The demo comp (`examples/sdf.wbl`): a Grid seen through a glass shape —
/// a seven-point spline blob melted into a disc, wobbling in fBm — with its rim
/// stroked, a drop shadow, a glow and ripples running out from it. Two
/// Sliders show on the node: `wobble` and `glow`.
#[cfg(test)]
pub fn demo() -> EditorGraph {
    use engine::graph::{GridParams, SdfParams, SplineMode, SplinePoint};
    let mut g = SdfGraph::default();
    let mut add = |id: u64, n: SdfNode, at: [f32; 2]| {
        g.nodes.insert(id, n);
        g.layout.insert(id, at);
    };
    // A lumpy blob left of centre, in FIELD space (y in width units:
    // the frame is 0.5625 tall).
    let (cx, cy) = (0.36f32, 0.28f32);
    let blob: Vec<SplinePoint> = (0..7)
        .map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 7.0 + 0.3;
            let r = [0.12, 0.085, 0.11, 0.075, 0.125, 0.09, 0.1][k];
            SplinePoint::new([cx + a.cos() * r, cy + a.sin() * r], 0.0)
        })
        .collect();
    let white = [1.0, 1.0, 1.0, 1.0];
    add(1, SdfNode::Spline { points: blob, mode: SplineMode::CatmullRom, closed: true, offset: [0.0, 0.0] }, [0.0, 0.0]);
    add(2, SdfNode::Circle { center: [1140.0, 560.0], radius: 160.0 }, [160.0, 0.0]);
    add(3, SdfNode::Combine { op: CombineOp::Union, smooth: 320.0 }, [80.0, 110.0]);
    add(4, SdfNode::Displace { amount: 8.0, size: 140.0, seed: 3.0, speed: 0.02, octaves: 3 }, [80.0, 220.0]);
    add(5, SdfNode::Glow { color: [0.35, 0.55, 1.0, 1.0], radius: 70.0, intensity: 0.9 }, [-240.0, 340.0]);
    add(6, SdfNode::Ripples {
        color: [0.6, 0.8, 1.0, 0.5],
        spacing: 30.0,
        width: 2.0,
        speed: 1.5,
        falloff: 240.0,
        side: Side::Outside,
    }, [-120.0, 340.0]);
    add(7, SdfNode::Shadow { color: [0.0, 0.0, 0.0, 0.7], offset: [18.0, 24.0], soft: 40.0 }, [0.0, 340.0]);
    add(8, SdfNode::Refract { amount: 28.0, width: 60.0 }, [120.0, 340.0]);
    add(9, SdfNode::Stroke { color: white, width: 3.0, soft: 0.0 }, [240.0, 340.0]);
    add(10, SdfNode::Merge { op: MergeOp::Over, mix: 1.0 }, [-180.0, 460.0]);
    add(11, SdfNode::Merge { op: MergeOp::Over, mix: 1.0 }, [-60.0, 560.0]);
    add(12, SdfNode::Merge { op: MergeOp::Over, mix: 1.0 }, [60.0, 660.0]);
    add(13, SdfNode::Merge { op: MergeOp::Over, mix: 1.0 }, [180.0, 760.0]);
    add(14, SdfNode::Output, [180.0, 880.0]);
    add(15, SdfNode::Const { value: 8.0, name: "wobble".into() }, [240.0, 160.0]);
    add(16, SdfNode::Const { value: 70.0, name: "glow".into() }, [-360.0, 220.0]);
    for (dst, port, src) in [
        (3, 0, 1),
        (3, 1, 2),
        (4, 0, 3),
        (4, 1, 15),
        (5, 0, 4),
        (5, 1, 16),
        (6, 0, 4),
        (7, 0, 4),
        (8, 0, 4),
        (9, 0, 4),
        (10, 0, 6),
        (10, 1, 5),
        (11, 0, 7),
        (11, 1, 10),
        (12, 0, 8),
        (12, 1, 11),
        (13, 0, 9),
        (13, 1, 12),
        (14, 0, 13),
    ] {
        g.edges.insert((dst, port), src);
    }
    g.next_id = 17;
    let mut comp = EditorGraph::default();
    let grid = comp.add(
        NodeKind::Grid(GridParams {
            spacing: [48.0, 48.0],
            width: 2.0,
            line: [0.16, 0.18, 0.24, 1.0],
            back: [0.015, 0.02, 0.035, 1.0],
            ..Default::default()
        }),
        egui::pos2(60.0, 40.0),
    );
    let sdf = comp.add(NodeKind::Sdf(SdfParams { graph: g, ..Default::default() }), egui::pos2(60.0, 160.0));
    comp.edges.push((sdf, 0, grid));
    comp.view = Some(sdf);
    comp
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `examples/sdf.wbl` is regenerated here, like the plant: it loads
    /// through the app's own path, compiles, and shows its two Sliders.
    #[test]
    fn sdf_demo_is_written_loads_and_compiles() {
        let g = demo();
        let sdf = g.view.unwrap();
        let empty_dirs: Vec<std::path::PathBuf> = Vec::new();
        let huds: Vec<widgets::HudSlider> = Vec::new();
        let activated = vec![sdf];
        let pins: Vec<u64> = Vec::new();
        let layouts: Vec<crate::PropLayout> = Vec::new();
        let s = crate::SetupSave {
            version: 4,
            fps: 24.0,
            pinned_dirs: &empty_dirs,
            graph: &g,
            canvas: None,
            range: Some((1, 120)),
            handles: (0, 0),
            play_range: None,
            view_range: None,
            loop_span: crate::LoopSpan::InOut,
            panel: Some((1, 1)),
            huds: &huds,
            activated: &activated,
            pinned_gizmos: &pins,
            prop_layouts: &layouts,
            patch_tiles: Vec::new(),
            chan_sets: &[],
        };
        let json = serde_json::to_string_pretty(&s).unwrap();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/sdf.wbl");
        if std::fs::read_to_string(&path).ok().as_deref() != Some(json.as_str()) {
            std::fs::write(&path, &json).unwrap();
        }
        let mut val: serde_json::Value = serde_json::from_str(&json).unwrap();
        crate::migrate_setup_json(&mut val);
        let l: crate::SetupLoad = serde_json::from_value(val).unwrap();
        let plan = engine::codegen::generate(&l.graph.to_engine_graph());
        assert!(plan.main.wgsl.contains("fn sde"), "the SDF entry is in the shader");
        assert!(plan.main.wgsl.contains("fn sdd"), "and the refraction's displacement");
        let Some(NodeKind::Sdf(p)) = l.graph.node(sdf).map(|n| &n.kind) else { panic!("gone") };
        let names: Vec<String> = p.graph.exposed().into_iter().map(|(n, _)| n).collect();
        assert_eq!(names, ["wobble", "glow"]);
    }

    #[test]
    fn a_graph_survives_the_editor_round_trip() {
        let mut g = starter();
        g.edges.insert((2, 0), 1);
        let back = from_editor(&to_editor(&g));
        assert_eq!(back.nodes, g.nodes);
        assert_eq!(back.edges, g.edges);
        assert_eq!(back.layout, g.layout);
        assert_eq!(back.next_id, g.next_id);
    }
}

/// Probe box half-size, in output pixels.
const PROBE: u32 = 4;

impl crate::Gfx {
    /// Mean r, g, b over a small box of the comp output at a fraction of
    /// its size.
    fn sdf_probe(&self, fx: f32, fy: f32) -> Option<[f32; 3]> {
        let (ow, oh) = self.comp.output_size()?;
        let x = ((fx * ow as f32) as u32).saturating_sub(PROBE).min(ow - 2 * PROBE);
        let y = ((fy * oh as f32) as u32).saturating_sub(PROBE).min(oh - 2 * PROBE);
        let c = |ch| self.probe_output_box(x, y, 2 * PROBE, 2 * PROBE, ch);
        Some([c(0)?, c(1)?, c(2)?])
    }

    /// --test-sdf driver: the SDF authoring loop through the app's own
    /// paths. Wire an SDF node over the viewed plate and open it (the
    /// starter graph comes up with its Spline selected); S-click three
    /// points into the viewer; the triangle renders filled white over the
    /// plate; viewing the Spline inside shows its DISTANCE as a map (violet
    /// inside, not white); stepping out keeps the drawing and the comp
    /// renders it. Pixel readings are believed only once two agree, and
    /// every wait is on the wall clock.
    pub(crate) fn drive_test_sdf(&mut self, raw: &mut egui::RawInput) {
        use crate::{DiveKind, NodeKind};
        static T0: std::sync::OnceLock<web_time::Instant> = std::sync::OnceLock::new();
        let t0 = *T0.get_or_init(web_time::Instant::now);
        let Some((stage, mut frames, sid)) = self.test.sdf else { return };
        frames += 1;
        let fail = |why: String| -> ! {
            println!("SDF TEST FAIL: {why}");
            std::process::exit(1);
        };
        if t0.elapsed().as_secs() > 90 {
            fail(format!(
                "timeout at stage {stage} (last reading {:?}, waiting to leave {:?})",
                self.test.sdf_last.map(|(_, v)| v),
                self.test.sdf_base
            ));
        }
        // S held lays a point down (a plain click is a selection): S goes
        // down with the press and comes up a frame after the release, since
        // the click resolves on the release.
        let key_s = |raw: &mut egui::RawInput, pressed: bool| {
            raw.events.push(egui::Event::Key {
                key: egui::Key::S,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let add = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            if down {
                key_s(raw, true);
            }
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        const SPOTS: [(f32, f32); 3] = [(0.3, 0.3), (0.7, 0.35), (0.5, 0.7)];
        // The triangle's centroid, where the fill is probed.
        const MID: (f32, f32) = (0.5, 0.45);
        let spot = |s: &Self, k: usize| -> egui::Pos2 {
            let r = s.test.rect;
            let f = SPOTS[k];
            egui::pos2(r.min.x + r.width() * f.0, r.min.y + r.height() * f.1)
        };
        let points = |s: &Self| -> Vec<engine::graph::SplinePoint> {
            s.egraph
                .nodes
                .values()
                .find_map(|n| match &n.kind {
                    NodeKind::SdfNode(SdfNode::Spline { points, .. }) => Some(points.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        };
        // A reading is believed once it has LEFT the picture the stage
        // started from and the same one comes back twice, 400 ms apart on
        // the wall clock: an edit recompiles (asynchronously, and a Metal
        // compile is ~100 ms), then renders, then shows — and under test
        // the loop is unpaced, so a frame count is not a duration.
        let settled = |s: &mut Self, from: Option<[f32; 3]>| -> Option<[f32; 3]> {
            let now_t = web_time::Instant::now();
            if s.test.sdf_last.is_some_and(|(t, _)| now_t.duration_since(t).as_millis() < 400) {
                return None;
            }
            let now = s.sdf_probe(MID.0, MID.1)?;
            let prev = s.test.sdf_last.replace((now_t, now)).map(|(_, v)| v);
            let same = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.01);
            match prev {
                Some(p) if same(p, now) && !from.is_some_and(|f| same(f, now)) => Some(now),
                _ => None,
            }
        };
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if !self.layout_settled() {
                    return;
                }
                self.test.rect = self.image_rect;
                let Some(plate) = self.egraph.active_view() else {
                    fail("nothing is viewed to draw over".into());
                };
                let id = self.egraph.add(
                    NodeKind::Sdf(Default::default()),
                    egui::Pos2::new(320.0, 40.0),
                );
                self.egraph.connect(id, 0, plate);
                self.egraph.view = Some(id);
                self.selected = vec![id];
                self.playing = false;
                self.playback_dirty = true;
                if let Err(e) = self.dive_into(id) {
                    fail(format!("could not open the node: {e}"));
                }
                self.test.sdf = Some((1, 0, id));
                return;
            }
            1 if frames > 2 => {
                if self.dive.last().map(|d| d.kind) != Some(DiveKind::Sdf) {
                    fail("the dive did not open an SDF graph".into());
                }
                let kinds: Vec<&'static str> = self
                    .egraph
                    .nodes
                    .values()
                    .filter_map(|n| match &n.kind {
                        NodeKind::SdfNode(k) => Some(k.name()),
                        _ => None,
                    })
                    .collect();
                if kinds != ["Spline", "Fill", "Output"] || self.selected != [1] {
                    fail(format!(
                        "an empty node should open as Spline -> Fill -> Output with the Spline \
                         picked: {kinds:?}, selected {:?}",
                        self.selected
                    ));
                }
                // The plate, before anything is drawn: what the fill must
                // move away from.
                if let Some(c) = settled(self, None) {
                    self.test.sdf_base = Some(c);
                    self.test.sdf_last = None;
                    next = 2;
                }
            }
            // Three S-clicks, each retried if it missed (the geo suite's
            // pattern: press, release, wait for the point).
            2 | 4 | 6 if frames > 6 => {
                let k = (stage as usize - 2) / 2;
                if points(self).len() > k {
                    next = stage + 2;
                } else {
                    add(raw, spot(self, k), true);
                    next = stage + 1;
                }
            }
            3 | 5 | 7 => {
                let k = (stage as usize - 3) / 2;
                if frames == 1 {
                    add(raw, spot(self, k), false);
                } else if frames == 2 {
                    key_s(raw, false);
                } else if points(self).len() > k {
                    next = stage + 1;
                } else if frames > 20 {
                    next = stage - 1;
                }
            }
            8 => {
                let p = points(self);
                if p.len() != 3 {
                    fail(format!("clicked 3 points, the spline has {}", p.len()));
                }
                for (q, (w, _)) in p.iter().zip(SPOTS) {
                    if (q.p[0] - w).abs() > 0.06 {
                        fail(format!("a point landed at x {:.3}, wanted {w:.2}", q.p[0]));
                    }
                }
                self.test.sdf_last = None;
                next = 9;
            }
            // The preview: the triangle, filled white over the plate.
            9 => {
                let from = self.test.sdf_base;
                if let Some(c) = settled(self, from) {
                    if c.iter().any(|v| (v - 1.0).abs() > 0.02) {
                        fail(format!("inside the drawn triangle reads {c:?}, wanted white"));
                    }
                    // Now LOOK at the Spline: its distance, as a map.
                    self.egraph.view = Some(1);
                    self.test.sdf_base = Some(c);
                    self.test.sdf_last = None;
                    next = 10;
                }
            }
            10 => {
                let from = self.test.sdf_base;
                if let Some(c) = settled(self, from) {
                    if !(c[2] > c[0] + 0.1 && c[0] < 0.8) {
                        fail(format!("viewing the Spline should show its inside violet: {c:?}"));
                    }
                    self.egraph.view = None;
                    if let Err(e) = self.dive_out() {
                        fail(format!("could not close the node: {e}"));
                    }
                    self.egraph.view = Some(sid);
                    self.playback_dirty = true;
                    self.test.sdf_base = Some(c);
                    self.test.sdf_last = None;
                    next = 11;
                }
            }
            // Out in the comp: the drawing came back with the node, and the
            // comp itself renders it.
            11 => {
                let Some(NodeKind::Sdf(p)) = self.egraph.node(sid).map(|n| &n.kind) else {
                    fail("the SDF node is gone".into());
                };
                let n = p.graph.nodes.len();
                let spline = p.graph.nodes.values().find_map(|k| match k {
                    SdfNode::Spline { points, .. } => Some(points.len()),
                    _ => None,
                });
                if n != 3 || spline != Some(3) || p.graph.edges.len() != 2 {
                    fail(format!(
                        "the graph came back as {n} nodes / {} edges, spline {spline:?}",
                        p.graph.edges.len()
                    ));
                }
                let from = self.test.sdf_base;
                if let Some(c) = settled(self, from) {
                    if c.iter().any(|v| (v - 1.0).abs() > 0.02) {
                        fail(format!("out of the dive the comp reads {c:?} inside, wanted white"));
                    }
                    println!(
                        "SDF TEST PASS: opened onto the starter with its Spline picked, drew a \
                         triangle in the viewer, it filled white over the plate, viewing the \
                         Spline showed its distance map, and stepping out the comp renders it"
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test.sdf = Some((next, if next == stage { frames } else { 0 }, sid));
    }
}
