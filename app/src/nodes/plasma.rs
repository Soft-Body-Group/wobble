//! Plasma's app half: the four pages (star, loops, streamers, motion), the
//! presets, the channel binding and the glyph. The params, the particle
//! shader and the layer are `engine::nodes::plasma` / `engine::plasmagpu`.
//!
//! Every number is `plasma.<name>` on `Channel::Custom` (the Text node's
//! precedent), so the flat channel enum does not grow.

use crate::anim::{self, Channel};
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, PanelTab, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::PlasmaParams;
use engine::nodes::plasma::{MAX_COUNT, PREFIX, PRESETS};
use std::collections::BTreeSet;

/// Two tiles: the rgb row needs track to aim along.
pub const TILES: usize = 2;

pub fn tabs() -> Vec<PanelTab> {
    vec![
        PanelTab::new("star", "star"),
        PanelTab::new("loops", "loops"),
        PanelTab::new("streamers", "streamers"),
        PanelTab::new("motion", "motion"),
    ]
}

/// The channel of a Plasma number by its name after `plasma.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

pub fn dyn_channels() -> Vec<Channel> {
    PlasmaParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &PlasmaParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut PlasmaParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(n) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(n, v);
        }
    }
}

pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut PlasmaParams,
    tab: &str,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
) -> Option<HudSlider> {
    let node = rcx.node;
    let mut h: Option<HudSlider> = None;
    let mut take = |got: Option<HudSlider>| {
        if got.is_some() {
            h = got;
        }
    };
    // One row: label, range, channel name.
    macro_rules! row {
        ($label:expr, $field:expr, $lo:expr, $hi:expr, $name:expr) => {
            take(widgets::slider(ui, rcx, $label, &mut $field, $lo, $hi, &[chan($name)]))
        };
    }
    match tab {
        "loops" => {
            row!("loops", p.loops, 1.0, 8000.0, "loops");
            row!("strands", p.strands, 1.0, 64.0, "strands");
            row!("regions", p.regions, 1.0, 120.0, "regions");
            row!("cluster", p.cluster, 0.0, 1.0, "cluster");
            row!("region size", p.region_size, 0.0, 1.5, "region.size");
            row!("span", p.span, 0.01, 1.5, "span");
            row!("span ±", p.span_var, 0.0, 1.0, "span.var");
            row!("height", p.height, 0.0, 2.0, "height");
            row!("height ±", p.height_var, 0.0, 1.0, "height.var");
            row!("thickness", p.thickness, 0.0, 1.5, "thickness");
            row!("arcade", p.arcade, 0.0, 0.6, "arcade");
            row!("lean", p.lean, 0.0, 1.5, "lean");
            row!("feet", p.feet, 0.0, 12.0, "feet");
            row!("life", p.life, 0.0, 1000.0, "life");
            ui.weak("a loop is `strands` nested field lines between two feet; span and");
            ui.weak("arcade are angles on the surface, height is in radii; life is the");
            ui.weak("frames before a loop fades and is reborn elsewhere (0 = forever)");
        }
        "streamers" => {
            row!("streamers", p.streamers, 0.0, 1.0, "streamers");
            row!("reach", p.reach, 0.0, 6.0, "reach");
            row!("falloff", p.falloff, 0.0, 12.0, "falloff");
            row!("pinch", p.pinch, 0.0, 1.0, "pinch");
            ui.weak("the share of the particles on OPEN lines, which leave the star:");
            ui.weak("reach in radii, falloff how fast their light dies along the way;");
            ui.weak("they root in the same regions the loops do, and pinch bends a");
            ui.weak("region's lines together into a helmet with a cusp");
        }
        "motion" => {
            row!("flow px", p.flow, -12.0, 12.0, "flow");
            row!("pulse", p.pulse, 0.0, 1.0, "pulse");
            row!("streak px", p.streak, 0.0, 64.0, "streak");
            row!("width px", p.width, 0.3, 12.0, "width");
            row!("fuzz px", p.fuzz, 0.0, 12.0, "fuzz");
            row!("turbulence", p.turbulence, 0.0, 80.0, "turbulence");
            row!("turb scale", p.turb_scale, 0.1, 24.0, "turb.scale");
            row!("turb speed", p.turb_speed, 0.0, 0.3, "turb.speed");
            row!("spin °", p.spin, -2.0, 2.0, "spin");
            ui.weak("flow is pixels a frame along the line; turbulence pushes the lines");
            ui.weak("(pixels) and is nothing at their feet · solved per frame from the");
            ui.weak("frame number: nothing is simulated, so scrubbing is live");
        }
        _ => {
            widgets::gen_format_row(ui, ("plasma-format", node), &mut p.size, &mut p.inherit);
            ui.horizontal(|ui| {
                ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("preset").small()));
                for pre in &PRESETS {
                    if ui.small_button(pre.name).on_hover_text(pre.hint).clicked() {
                        p.apply_preset(pre);
                    }
                }
            });
            let [cx, cy] = &mut p.center;
            take(widgets::drags(
                ui,
                rcx,
                "centre",
                &mut [cx, cy],
                1.0,
                (-4096.0, 8192.0),
                &[chan("center.x"), chan("center.y")],
            ));
            row!("radius px", p.radius, 1.0, 4000.0, "radius");
            row!("particles", p.count, 0.0, MAX_COUNT, "count");
            row!("seed", p.seed, 0.0, 100.0, "seed");
            row!("depth", p.depth, 0.0, 1.0, "depth");
            row!("brightness", p.brightness, 0.0, 10.0, "brightness");
            {
                let key = "plasma-color";
                let mut gang = !ungang.contains(key);
                take(widgets::rgb_row(
                    ui,
                    rcx,
                    "colour",
                    &mut p.color,
                    &mut gang,
                    RgbSpec::scale(0.0, 4.0, working.primaries.luma_weights())
                        .with_display(primaries_matrix(working.primaries, Primaries::Rec709)),
                    &["color.r", "color.g", "color.b"].map(chan),
                ));
                if gang {
                    ungang.remove(key);
                } else {
                    ungang.insert(key.to_string());
                }
            }
            row!("body", p.body, 0.0, 4.0, "body");
            row!("limb", p.limb, -2.0, 1.0, "limb");
            row!("mottle", p.mottle, 0.0, 1.0, "mottle");
            row!("mottle scale", p.mottle_scale, 1.0, 60.0, "mottle.scale");
            row!("mottle speed", p.mottle_speed, 0.0, 0.2, "mottle.speed");
            row!("alpha", p.alpha, 0.0, 20.0, "alpha");
            ui.checkbox(&mut p.occlude, egui::RichText::new("the disc hides what is behind it").small())
                .on_hover_text("field lines are rooted on a sphere: off, the far side shows through the disc");
            ui.checkbox(&mut p.eclipse, egui::RichText::new("eclipse: a black disc in front").small())
                .on_hover_text("the moon: only what stands clear of the limb shows, and the body is not drawn");
            ui.weak("depth 0 keeps every loop on the limb (flat, side-on); 1 spreads them");
            ui.weak("over the sphere · body draws the star's disc (limb below 0 brightens");
            ui.weak("its edge, mottle is its granulation) · the light is additive: alpha 0");
            ui.weak("is pure emission");
        }
    }
    h
}

/// The graph glyph: a disc low in the tile with two loops arching off it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let centre = Pos2::new(c.x, c.y + r * 0.45);
    let rad = r * 0.62;
    painter.circle(centre, rad, fill, stroke);
    let line = Stroke::new((r * 0.11).max(1.0), fill);
    // An arch between two limb angles (radians, 0 = up), `h` radii tall.
    let arch = |a0: f32, a1: f32, h: f32| {
        let pts: Vec<Pos2> = (0..=10)
            .map(|i| {
                let s = i as f32 / 10.0;
                let a = a0 + (a1 - a0) * s;
                let k = rad * (1.0 + h * (std::f32::consts::PI * s).sin());
                Pos2::new(centre.x + a.sin() * k, centre.y - a.cos() * k)
            })
            .collect();
        painter.add(egui::Shape::line(pts, line));
    };
    arch(-0.75, 0.05, 0.75);
    arch(0.25, 0.9, 0.45);
}

/// The demo comp (`examples/corona.wbl`): a star's corona out of three
/// Plasma nodes added up — the surface (the disc under a carpet of tiny
/// loops), the streamers, and the active regions' loop arcades on top —
/// with a Glow over the lot. Press play.
#[cfg(test)]
pub fn demo() -> crate::editor::EditorGraph {
    use engine::graph::{BlendOp, GlowParams, NodeKind};
    let layer = |name: &str, seed: f32| {
        let mut p = PlasmaParams { seed, ..Default::default() };
        p.apply_preset(PRESETS.iter().find(|x| x.name == name).unwrap());
        NodeKind::Plasma(p)
    };
    let add = || NodeKind::Blend { op: BlendOp::Add, mix: 1.0, gain: [1.0, 1.0] };
    let mut comp = crate::editor::EditorGraph::default();
    let surface = comp.add(layer("surface", 0.0), egui::pos2(60.0, 40.0));
    let streamers = comp.add(layer("streamers", 1.0), egui::pos2(260.0, 40.0));
    let regions = comp.add(layer("active regions", 2.0), egui::pos2(460.0, 40.0));
    let a = comp.add(add(), egui::pos2(160.0, 180.0));
    comp.edges.push((a, 0, streamers));
    comp.edges.push((a, 1, surface));
    let b = comp.add(add(), egui::pos2(310.0, 300.0));
    comp.edges.push((b, 0, regions));
    comp.edges.push((b, 1, a));
    let glow = comp.add(
        NodeKind::Glow(GlowParams { size: 40.0, threshold: 0.3, intensity: 0.6, ..Default::default() }),
        egui::pos2(310.0, 420.0),
    );
    comp.edges.push((glow, 0, b));
    comp.view = Some(glow);
    comp
}

/// The second demo (`examples/eclipse.wbl`): the white-light corona of
/// a total eclipse — fine soft rays gathered into helmets behind a black
/// disc — with a few pink prominences on the limb and a Glow.
#[cfg(test)]
pub fn demo_eclipse() -> crate::editor::EditorGraph {
    use engine::graph::{BlendOp, GlowParams, NodeKind};
    let layer = |name: &str, seed: f32| {
        let mut p = PlasmaParams { seed, radius: 190.0, ..Default::default() };
        p.apply_preset(PRESETS.iter().find(|x| x.name == name).unwrap());
        NodeKind::Plasma(p)
    };
    let mut comp = crate::editor::EditorGraph::default();
    let corona = comp.add(layer("eclipse", 0.0), egui::pos2(60.0, 40.0));
    let pink = comp.add(layer("prominences", 1.0), egui::pos2(260.0, 40.0));
    let add = comp.add(NodeKind::Blend { op: BlendOp::Add, mix: 1.0, gain: [1.0, 1.0] }, egui::pos2(160.0, 180.0));
    comp.edges.push((add, 0, pink));
    comp.edges.push((add, 1, corona));
    let glow = comp.add(
        NodeKind::Glow(GlowParams { size: 50.0, falloff: 0.8, threshold: 0.0, knee: 0.0, intensity: 0.5, ..Default::default() }),
        egui::pos2(160.0, 300.0),
    );
    comp.edges.push((glow, 0, add));
    comp.view = Some(glow);
    comp
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// `examples/corona.wbl` and `eclipse.wbl` are regenerated here, like
    /// the plant: each loads through the app's own path and compiles to a
    /// plan reading one particle layer per Plasma.
    #[test]
    fn corona_demo_is_written_loads_and_compiles() {
        written("corona", demo(), 3);
        written("eclipse", demo_eclipse(), 2);
    }

    fn written(name: &str, g: crate::editor::EditorGraph, want: usize) {
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
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../examples/{name}.wbl"));
        if std::fs::read_to_string(&path).ok().as_deref() != Some(json.as_str()) {
            std::fs::write(&path, &json).unwrap();
        }
        let mut val: serde_json::Value = serde_json::from_str(&json).unwrap();
        crate::migrate_setup_json(&mut val);
        let l: crate::SetupLoad = serde_json::from_value(val).unwrap();
        let eg = l.graph.to_engine_graph();
        let plasmas = eg.nodes.values().filter(|k| matches!(k, NodeKind::Plasma(_))).count();
        assert_eq!(plasmas, want);
        let plan = engine::codegen::generate(&eg);
        let layers = plan
            .main
            .sources
            .iter()
            .chain(plan.blurs.iter().flat_map(|b| b.2.sources.iter()))
            .filter(|s| matches!(s, engine::codegen::SourceBinding::Plasma(_)))
            .count();
        assert_eq!(layers, want, "each Plasma is one layer read");
    }

    #[test]
    fn every_channel_reads_back_what_it_set() {
        let mut p = PlasmaParams::default();
        for (i, ch) in dyn_channels().into_iter().enumerate() {
            let v = 0.5 + i as f32;
            set(&mut p, ch, v);
            assert_eq!(get(&p, ch), Some(v));
        }
    }
}
