//! Particles' app half: the owner's panel and presets, the panel of every
//! node inside its graph, the round trip between a `ParticleGraph` and the
//! editor graph a dive swaps in, the glyph and the demo comp. The params,
//! the graph and the simulation are `engine::nodes::particles` /
//! `engine::particlegpu`.
//!
//! The owner's numbers are `particles.<name>` on `Channel::Custom` (the
//! Plasma precedent). The numbers INSIDE the graph are authored data, not
//! channels: a dive drops inner curves (as an SDF's does), so offering a
//! key dot there would be a promise the dive breaks.

use crate::anim::{self, Channel};
use crate::editor::{self, EditorGraph};
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::{NodeKind, PNode, ParticlesParams};
use engine::nodes::particles::{
    EmitParams, EmitShape, ForceKind, ForceParams, ParticleGraph, RenderParams, PREFIX, PRESETS,
};

pub const TILES: usize = 2;

pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

pub fn dyn_channels() -> Vec<Channel> {
    ParticlesParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &ParticlesParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut ParticlesParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(n) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(n, v);
        }
    }
}

/// A particle graph as the node editor holds it: every node a
/// `NodeKind::ParticleNode`, so selection, wiring and undo work unchanged.
pub fn to_editor(f: &ParticleGraph) -> EditorGraph {
    let mut g = EditorGraph::default();
    for (i, (id, node)) in f.nodes.iter().enumerate() {
        let p = f.layout.get(id).copied().unwrap_or([120.0 * i as f32, 60.0 * i as f32]);
        g.nodes.insert(*id, editor::ENode::new(NodeKind::ParticleNode(node.clone()), egui::pos2(p[0], p[1])));
    }
    for ((dst, port), src) in &f.edges {
        g.edges.push((*dst, *port, *src));
    }
    g.next_id = f.next_id.max(f.nodes.keys().copied().max().unwrap_or(0) + 1);
    g
}

/// The other direction: anything that is not a particle node is dropped.
pub fn from_editor(g: &EditorGraph) -> ParticleGraph {
    let mut f = ParticleGraph::default();
    for (id, n) in &g.nodes {
        if let NodeKind::ParticleNode(k) = &n.kind {
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

/// The owner's panel: format, timing, quality, presets.
pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut ParticlesParams) -> Option<HudSlider> {
    let node = rcx.node;
    let mut h: Option<HudSlider> = None;
    let mut take = |got: Option<HudSlider>| {
        if got.is_some() {
            h = got;
        }
    };
    macro_rules! row {
        ($label:expr, $field:expr, $lo:expr, $hi:expr, $name:expr) => {
            take(widgets::slider(ui, rcx, $label, &mut $field, $lo, $hi, &[chan($name)]))
        };
    }
    widgets::gen_format_row(ui, ("particles-format", node), &mut p.size, &mut p.inherit);
    ui.horizontal_wrapped(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("preset").small()));
        for pre in &PRESETS {
            if ui.small_button(pre.name).on_hover_text(pre.hint).clicked() {
                p.graph = (pre.build)();
            }
        }
    });
    row!("start", p.start, -100.0, 1000.0, "start");
    row!("preroll", p.preroll, 0.0, 1000.0, "preroll");
    row!("substeps", p.substeps, 1.0, 8.0, "substeps");
    row!("quality", p.quality, 0.1, 8.0, "quality");
    row!("seed", p.seed, 0.0, 100.0, "seed");
    row!("alpha", p.alpha, 0.0, 20.0, "alpha");
    let c = p.compile(1.0);
    ui.weak(format!(
        "{} slots · {} draw{} · open it (↩) to edit the emitters, forces and renders",
        c.slots,
        c.renders.len(),
        if c.renders.len() == 1 { "" } else { "s" }
    ));
    ui.weak("simulated on the GPU from `start - preroll`; scrubbing restores the");
    ui.weak("nearest checkpoint (every 8 frames) and steps forward · quality");
    ui.weak("multiplies every rate and divides the light, so it is not an exposure");
    h
}

fn shape_row(ui: &mut egui::Ui, id: u64, s: &mut EmitShape) {
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("shape").small()));
        egui::ComboBox::from_id_salt(("pemit-shape", id)).selected_text(s.name()).show_ui(ui, |ui| {
            for k in EmitShape::ALL {
                ui.selectable_value(s, k, k.name());
            }
        });
    });
}

fn kind_row(ui: &mut egui::Ui, id: u64, s: &mut ForceKind) {
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("force").small()));
        egui::ComboBox::from_id_salt(("pforce-kind", id)).selected_text(s.name()).show_ui(ui, |ui| {
            for k in ForceKind::ALL {
                ui.selectable_value(s, k, k.name());
            }
        });
    });
}

/// One node INSIDE a particle graph.
pub fn node_panel(ui: &mut egui::Ui, rcx: &mut RowCtx, node: u64, n: &mut PNode, working: Colorspace) {
    macro_rules! row {
        ($label:expr, $field:expr, $lo:expr, $hi:expr) => {
            let _ = widgets::slider(ui, rcx, $label, &mut $field, $lo, $hi, &[]);
        };
    }
    macro_rules! xy {
        ($label:expr, $v:expr) => {{
            let [x, y] = &mut $v;
            let _ = widgets::drags(ui, rcx, $label, &mut [x, y], 1.0, (-4096.0, 8192.0), &[]);
        }};
    }
    match n {
        PNode::Emit(e) => emit_panel(ui, rcx, node, e),
        PNode::Force(f) => {
            kind_row(ui, node, &mut f.kind);
            let k = f.kind;
            match k {
                ForceKind::Gravity => {
                    row!("strength", f.strength, -2.0, 2.0);
                    row!("direction °", f.direction, -180.0, 180.0);
                    ui.weak("a constant pull, px a frame a frame; -90 is down");
                }
                ForceKind::Radial | ForceKind::Vortex => {
                    row!("strength", f.strength, -2.0, 2.0);
                    xy!("centre", f.center);
                    row!("radius px", f.radius, 0.0, 4000.0);
                    ui.weak(if k == ForceKind::Radial {
                        "pushes away from the centre (negative pulls in), fading to nothing at the radius (0 = everywhere)"
                    } else {
                        "spins round the centre, counter-clockwise for a positive strength; radius 0 = everywhere"
                    });
                }
                ForceKind::Drag => {
                    row!("drag", f.strength, 0.0, 1.0);
                    ui.weak("the share of its speed a particle loses each frame");
                }
                ForceKind::Turbulence => {
                    row!("strength", f.strength, 0.0, 20.0);
                    row!("scale px", f.scale, 4.0, 2000.0);
                    row!("speed", f.speed, 0.0, 0.5);
                    row!("follow", f.follow, 0.0, 1.0);
                    row!("seed", f.seed, 0.0, 100.0);
                    ui.weak("curl noise: swirls that neither gather nor spread · follow 1 carries");
                    ui.weak("particles along (px a frame), 0 pushes them (px a frame a frame)");
                }
                ForceKind::FieldLines => {
                    row!("speed", f.strength, -10.0, 10.0);
                    row!("follow", f.follow, 0.0, 1.0);
                    xy!("centre", f.center);
                    row!("radius px", f.radius, 1.0, 4000.0);
                    row!("pairs", f.pairs, 0.0, 256.0);
                    row!("regions", f.regions, 1.0, 64.0);
                    row!("cluster", f.cluster, 0.0, 1.0);
                    row!("region size", f.region_size, 0.0, 1.5);
                    row!("span", f.span, 0.01, 1.5);
                    row!("span ±", f.span_var, 0.0, 1.0);
                    row!("depth", f.depth, 0.0, 0.5);
                    row!("open", f.open, 0.0, 1.0);
                    row!("foot px", f.foot, 0.0, 60.0);
                    row!("strands", f.strands, 0.0, 64.0);
                    row!("fan °", f.fan, 0.0, 180.0);
                    row!("fill", f.fill, 0.0, 1.0);
                    row!("sphere", f.sphere, 0.0, 1.0);
                    ui.checkbox(&mut f.own_pair, egui::RichText::new("each line feels only its own pair").small())
                        .on_hover_text("clean arcs and straight rays however crowded the ring; off, every pole pulls and the lines tangle");
                    row!("seed", f.seed, 0.0, 100.0);
                    ui.checkbox(&mut f.both_ways, egui::RichText::new("flow both ways along a line").small());
                    ui.weak("± pole pairs just under a ring; particles ride the field between");
                    ui.weak("them — loops — and off the unpaired ones (open) — streamers · foot > 0");
                    ui.weak("re-homes every birth onto a pole, so loops are rooted bundles: strands");
                    ui.weak("lines a pole (0 = a haze), leaving within fan° of the line to its");
                    ui.weak("partner (taller loops as it widens) · fill puts flat pairs on the face");
                    ui.weak("sphere 0 keeps every pole on the ring (flat, side-on); 1 spreads them");
                    ui.weak("over a sphere seen from the front — give the Render the same disc and");
                    ui.weak("tick its \"hide what is behind the disc\"");
                }
                ForceKind::Kill => {
                    xy!("centre", f.center);
                    row!("radius px", f.radius, 0.0, 4000.0);
                    ui.checkbox(&mut f.invert, egui::RichText::new("kill outside instead").small());
                }
            }
            if ui.small_button("reset to this kind's defaults").clicked() {
                *f = ForceParams::of(k);
            }
        }
        PNode::Merge => {
            ui.weak("joins two streams: the forces after it act on both");
        }
        PNode::Render(r) => render_panel(ui, rcx, r, working),
    }
}

fn emit_panel(ui: &mut egui::Ui, rcx: &mut RowCtx, node: u64, e: &mut EmitParams) {
    macro_rules! row {
        ($label:expr, $field:expr, $lo:expr, $hi:expr) => {
            let _ = widgets::slider(ui, rcx, $label, &mut $field, $lo, $hi, &[]);
        };
    }
    shape_row(ui, node, &mut e.shape);
    {
        let [x, y] = &mut e.center;
        let _ = widgets::drags(ui, rcx, "centre", &mut [x, y], 1.0, (-4096.0, 8192.0), &[]);
    }
    match e.shape {
        EmitShape::Point => {}
        EmitShape::Line => {
            row!("length px", e.size[0], 0.0, 4000.0);
        }
        EmitShape::Ring | EmitShape::Disc => {
            row!("radius px", e.size[0], 0.0, 4000.0);
        }
        EmitShape::Box => {
            let [x, y] = &mut e.size;
            let _ = widgets::drags(ui, rcx, "size", &mut [x, y], 1.0, (0.0, 8192.0), &[]);
        }
    }
    if matches!(e.shape, EmitShape::Line | EmitShape::Box) {
        row!("angle °", e.angle, -180.0, 180.0);
    }
    row!("rate", e.rate, 0.0, 50_000.0);
    row!("life", e.life, 1.0, 1000.0);
    row!("life ±", e.life_var, 0.0, 1.0);
    row!("speed", e.speed, -50.0, 50.0);
    row!("speed ±", e.speed_var, 0.0, 1.0);
    row!("direction °", e.direction, -180.0, 180.0);
    row!("spread °", e.spread, 0.0, 360.0);
    row!("normal", e.normal, 0.0, 1.0);
    row!("spots", e.spots, 0.0, 64.0);
    row!("spot size", e.spot_size, 0.0, 0.5);
    row!("clump", e.clump, 0.0, 1.0);
    row!("seed", e.seed, 0.0, 100.0);
    ui.weak("rate is particles a frame, life and speed in frames and px a frame;");
    ui.weak("normal 1 launches out of the shape · spots gather births into hot");
    ui.weak("spots along it, clump is the share they take");
}

fn render_panel(ui: &mut egui::Ui, rcx: &mut RowCtx, r: &mut RenderParams, working: Colorspace) {
    macro_rules! row {
        ($label:expr, $field:expr, $lo:expr, $hi:expr) => {
            let _ = widgets::slider(ui, rcx, $label, &mut $field, $lo, $hi, &[]);
        };
    }
    let spec = || {
        RgbSpec::scale(0.0, 4.0, working.primaries.luma_weights())
            .with_display(primaries_matrix(working.primaries, Primaries::Rec709))
    };
    // Not channels (see the module docs), so the bare slider: ganged, and
    // the slider's own modifier ungangs it for a drag.
    for (label, v) in [("colour", &mut r.color), ("at death", &mut r.color_end)] {
        ui.horizontal(|ui| {
            ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new(label).small()));
            let mut gang = true;
            crate::rgbslider::show(ui, ("prender", rcx.node, label), v, &mut gang, spec(), None, rcx.hud_live);
        });
    }
    row!("brightness", r.brightness, 0.0, 10.0);
    row!("streak px", r.streak, 0.0, 64.0);
    row!("stretch", r.stretch, 0.0, 10.0);
    row!("width px", r.width, 0.3, 16.0);
    row!("fade in", r.fade_in, 0.0, 1.0);
    row!("fade out", r.fade_out, 0.0, 1.0);
    row!("vary", r.vary, 0.0, 1.0);
    ui.checkbox(&mut r.occlude, egui::RichText::new("hide what is behind the disc").small())
        .on_hover_text("for Field Lines on a sphere: particles on the far side, inside the disc's outline, are not drawn");
    if r.occlude {
        let [x, y] = &mut r.disc_center;
        let _ = widgets::drags(ui, rcx, "disc centre", &mut [x, y], 1.0, (-4096.0, 8192.0), &[]);
        row!("disc radius", r.disc_radius, 0.0, 4000.0);
    }
    ui.weak("streaks of light, added up: length = streak + stretch x speed;");
    ui.weak("the colour runs from birth to death and fades in and out over");
    ui.weak("those shares of a life");
}

/// The graph glyph: a spray of dots off a source.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let src = Pos2::new(c.x - r * 0.45, c.y + r * 0.45);
    painter.circle(src, r * 0.22, fill, stroke);
    let dots = [(0.1, -0.15, 0.13), (0.45, -0.05, 0.1), (0.25, -0.5, 0.11), (0.6, -0.45, 0.08), (-0.1, -0.6, 0.09)];
    for (dx, dy, s) in dots {
        painter.circle_filled(Pos2::new(c.x + dx * r, c.y + dy * r), r * s, fill);
    }
}

/// The demo comp (`examples/particles_corona.wbl`): the Plasma node's
/// corona rebuilt from the parts — a Particles node on the corona preset
/// (loops riding a field off pole pairs on the limb, streamers off the
/// unpaired ones), a black Planet-sized disc over it, and a Glow.
#[cfg(test)]
pub fn demo() -> EditorGraph {
    use engine::graph::{BlendOp, GlowParams};
    let mut comp = EditorGraph::default();
    let p = ParticlesParams {
        graph: PRESETS.iter().find(|x| x.name == "corona").map(|x| (x.build)()).unwrap(),
        preroll: 150.0,
        ..Default::default()
    };
    let parts = comp.add(NodeKind::Particles(p), egui::pos2(60.0, 40.0));
    let disc = engine::graph::PlasmaParams {
        count: 0.0,
        body: 0.0,
        eclipse: true,
        ..Default::default()
    };
    let moon = comp.add(NodeKind::Plasma(disc), egui::pos2(260.0, 40.0));
    let over = comp.add(NodeKind::Blend { op: BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] }, egui::pos2(160.0, 180.0));
    comp.edges.push((over, 0, moon));
    comp.edges.push((over, 1, parts));
    let glow = comp.add(
        NodeKind::Glow(GlowParams { size: 40.0, threshold: 0.0, knee: 0.0, intensity: 0.5, ..Default::default() }),
        egui::pos2(160.0, 300.0),
    );
    comp.edges.push((glow, 0, over));
    comp.view = Some(glow);
    comp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_channel_reads_back_what_it_set() {
        let mut p = ParticlesParams::default();
        for (i, ch) in dyn_channels().into_iter().enumerate() {
            let v = 0.5 + i as f32;
            set(&mut p, ch, v);
            assert_eq!(get(&p, ch), Some(v));
        }
    }

    #[test]
    fn the_dive_round_trip_keeps_the_graph() {
        for pre in PRESETS {
            let g = (pre.build)();
            let back = from_editor(&to_editor(&g));
            assert_eq!(back.nodes, g.nodes, "{}", pre.name);
            assert_eq!(back.edges, g.edges, "{}", pre.name);
            assert_eq!(back.layout, g.layout, "{}", pre.name);
        }
    }

    /// `examples/particles_corona.wbl` is regenerated here: it loads
    /// through the app's own path and compiles to a plan reading one
    /// particle layer.
    #[test]
    fn corona_demo_is_written_loads_and_compiles() {
        let g = demo();
        let empty_dirs: Vec<std::path::PathBuf> = Vec::new();
        let huds: Vec<widgets::HudSlider> = Vec::new();
        let activated: Vec<u64> = Vec::new();
        let pins: Vec<u64> = Vec::new();
        let layouts: Vec<crate::PropLayout> = Vec::new();
        let s = crate::SetupSave {
            version: 4,
            fps: 24.0,
            pinned_dirs: &empty_dirs,
            graph: &g,
            canvas: None,
            range: Some((1, 240)),
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
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/particles_corona.wbl");
        if std::fs::read_to_string(&path).ok().as_deref() != Some(json.as_str()) {
            std::fs::write(&path, &json).unwrap();
        }
        let mut val: serde_json::Value = serde_json::from_str(&json).unwrap();
        crate::migrate_setup_json(&mut val);
        let l: crate::SetupLoad = serde_json::from_value(val).unwrap();
        let eg = l.graph.to_engine_graph();
        let parts: Vec<&ParticlesParams> = eg
            .nodes
            .values()
            .filter_map(|k| if let NodeKind::Particles(p) = k { Some(p) } else { None })
            .collect();
        assert_eq!(parts.len(), 1);
        assert!(parts[0].compile(1.0).slots > 0, "the graph survived the save");
        let plan = engine::codegen::generate(&eg);
        let layers = plan
            .main
            .sources
            .iter()
            .chain(plan.blurs.iter().flat_map(|b| b.2.sources.iter()))
            .filter(|s| matches!(s, engine::codegen::SourceBinding::Particles(_)))
            .count();
        assert_eq!(layers, 1, "one particle layer read");
    }
}
