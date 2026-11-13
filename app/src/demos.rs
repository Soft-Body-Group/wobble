//! Demo setups built in code, so they are regenerated from ONE definition
//! whenever the nodes they use change, and a test proves each one still
//! loads, compiles and draws. `examples/<name>.wbl` is what the test
//! writes; open it with ⌘O.

use engine::field::{FieldGraph, FieldNode, MathOp, WarpMode};
use engine::geo::{GeoGraph, GeoNode, Pivot, ResampleMode, Style};
use engine::graph::{GeoParams, NodeKind, NoiseParams, SplineMode, SplinePoint};

use crate::editor::EditorGraph;

/// Where a node sits on the geo canvas: columns are stages, rows are
/// streams, so the dive reads left to right the way the plant grows.
fn at(col: f32, row: f32) -> [f32; 2] {
    [col * 200.0, row * 95.0]
}

fn pt(x: f32, y: f32) -> SplinePoint {
    SplinePoint::new([x, y], 0.0)
}

fn style(fill: [f32; 4], stroke: [f32; 4], width: f32) -> Style {
    Style { fill, stroke, width, ..Default::default() }
}

/// A procedural plant: a stem that sways in a Noise wind, branches
/// instanced along it on alternating sides, leaves instanced along the
/// branches, and a flower on top that sways with the same wind. Seven
/// controls are exposed on the 2dini node — wind, stem width, branch
/// spacing, branch length, leaf spacing, leaf size, bloom — and the wind
/// Noise animates on its own, so the setup moves when played.
///
/// The comp is two nodes: the wind Noise wired into the plant's field
/// port, and the plant, viewed.
pub fn plant() -> EditorGraph {
    // --- the geometry graph -------------------------------------------
    let mut g = GeoGraph::default();
    let mut id = 0u64;
    let mut add = |g: &mut GeoGraph, n: GeoNode, pos: [f32; 2]| -> u64 {
        id += 1;
        g.nodes.insert(id, n);
        g.layout.insert(id, pos);
        id
    };
    let green = [0.22, 0.58, 0.22, 1.0];
    let dark_green = [0.10, 0.34, 0.12, 1.0];
    let none = [0.0; 4];

    // Controls: the sliders that become widgets on the plant's panel.
    let wind = add(&mut g, GeoNode::Slider { value: 0.03, name: "wind".into() }, at(0.0, 0.0));
    let stem_width =
        add(&mut g, GeoNode::Slider { value: 7.0, name: "stem width".into() }, at(0.0, 1.0));
    let branch_spacing =
        add(&mut g, GeoNode::Slider { value: 70.0, name: "branch spacing".into() }, at(0.0, 2.0));
    let branch_length =
        add(&mut g, GeoNode::Slider { value: 1.0, name: "branch length".into() }, at(0.0, 3.0));
    let leaf_spacing =
        add(&mut g, GeoNode::Slider { value: 40.0, name: "leaf spacing".into() }, at(0.0, 4.0));
    let leaf_size =
        add(&mut g, GeoNode::Slider { value: 0.55, name: "leaf size".into() }, at(0.0, 5.0));
    let bloom = add(&mut g, GeoNode::Slider { value: 0.075, name: "bloom".into() }, at(0.0, 6.0));

    // The wind: the Noise on the field port, read twice with different
    // seed shifts so x and y sway independently.
    let wind_x = add(&mut g, GeoNode::FieldIn { shift: 0.0 }, at(1.0, 0.0));
    let wind_y = add(&mut g, GeoNode::FieldIn { shift: 11.0 }, at(1.0, 1.0));

    // Ground: a strip along the bottom (y is in width units; the canvas is
    // 16:9, so the bottom edge is at 0.5625).
    let ground = add(
        &mut g,
        GeoNode::Spline {
            points: vec![pt(0.0, 0.545), pt(1.0, 0.545), pt(1.0, 0.5625), pt(0.0, 0.5625)],
            mode: SplineMode::Linear,
            closed: true,
            style: style([0.12, 0.28, 0.12, 1.0], none, 0.0),
        },
        at(2.0, 5.0),
    );

    // Stem: an open spline, resampled finely so the wind bends it as a
    // curve rather than as four straight legs.
    let stem = add(
        &mut g,
        GeoNode::Spline {
            points: vec![pt(0.5, 0.545), pt(0.487, 0.44), pt(0.512, 0.32), pt(0.5, 0.20)],
            mode: SplineMode::CatmullRom,
            closed: false,
            style: style(none, green, 6.0),
        },
        at(2.0, 2.0),
    );
    let stem_fine = add(
        &mut g,
        GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 10.0, count: 32 },
        at(3.0, 2.0),
    );
    g.edges.insert((stem_fine, 0), stem);
    let stem_sway = add(
        &mut g,
        GeoNode::Displace { mode: WarpMode::Offset, amount: 0.03, x: 0.5, y: 0.5 },
        at(4.0, 2.0),
    );
    g.edges.insert((stem_sway, 0), stem_fine);
    g.edges.insert((stem_sway, 1), wind_x);
    g.edges.insert((stem_sway, 2), wind_y);
    g.edges.insert((stem_sway, 3), wind);
    let stem_style = add(
        &mut g,
        GeoNode::Style { style: style(none, green, 7.0) },
        at(5.0, 2.0),
    );
    g.edges.insert((stem_style, 0), stem_sway);
    g.edges.insert((stem_style, 9), stem_width);

    // Branches: a short curved stroke instanced along the swaying stem
    // (orient = the stem's tangent, so each branch leaves the stem at the
    // same angle), on both sides at different heights — two streams at
    // different spacings.
    let branch_pts_a = add(
        &mut g,
        GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 70.0, count: 32 },
        at(5.0, 3.0),
    );
    g.edges.insert((branch_pts_a, 0), stem_sway);
    g.edges.insert((branch_pts_a, 1), branch_spacing);
    let branch_spacing_b = add(
        &mut g,
        GeoNode::Math { op: MathOp::Mul, a: 70.0, b: 1.41 },
        at(4.0, 4.0),
    );
    g.edges.insert((branch_spacing_b, 0), branch_spacing);
    let branch_pts_b = add(
        &mut g,
        GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 99.0, count: 32 },
        at(5.0, 4.0),
    );
    g.edges.insert((branch_pts_b, 0), stem_sway);
    g.edges.insert((branch_pts_b, 1), branch_spacing_b);
    // The branch: an open curve from the origin along +x, lifting as it
    // goes, so an Instance about the origin hangs it off the stem by its
    // base and the rotation sets how steeply it climbs.
    let branch = add(
        &mut g,
        GeoNode::Spline {
            points: vec![pt(0.0, 0.0), pt(0.045, -0.01), pt(0.085, -0.028), pt(0.12, -0.052)],
            mode: SplineMode::CatmullRom,
            closed: false,
            style: style(none, green, 3.0),
        },
        at(4.0, 5.5),
    );
    // The other side gets the branch MIRRORED, so its lift is still a
    // lift: rotating one template both ways would droop one side.
    let branch_mirror = add(
        &mut g,
        GeoNode::Transform { translate: [0.0, 0.0], rotate: 0.0, scale: [1.0, -1.0], pivot: [0.0, 0.0], inherit: false },
        at(5.0, 5.5),
    );
    g.edges.insert((branch_mirror, 0), branch);
    let branches_a = add(
        &mut g,
        GeoNode::Instance { scale: 1.0, rotation: 1.15, inherit_cd: true, pivot: Pivot::Origin },
        at(6.0, 3.0),
    );
    g.edges.insert((branches_a, 0), branch);
    g.edges.insert((branches_a, 1), branch_pts_a);
    g.edges.insert((branches_a, 2), branch_length);
    let branches_b = add(
        &mut g,
        GeoNode::Instance { scale: 1.0, rotation: -1.15, inherit_cd: true, pivot: Pivot::Origin },
        at(6.0, 4.0),
    );
    g.edges.insert((branches_b, 0), branch_mirror);
    g.edges.insert((branches_b, 1), branch_pts_b);
    g.edges.insert((branches_b, 2), branch_length);
    let branches = add(&mut g, GeoNode::Merge { count: 2 }, at(7.0, 3.5));
    g.edges.insert((branches, 0), branches_a);
    g.edges.insert((branches, 1), branches_b);

    // Leaves: points along every branch (orient = the branch's tangent), a
    // leaf instanced on each, again two streams for the two sides.
    let leaf_pts_a = add(
        &mut g,
        GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 40.0, count: 32 },
        at(8.0, 3.0),
    );
    g.edges.insert((leaf_pts_a, 0), branches);
    g.edges.insert((leaf_pts_a, 1), leaf_spacing);
    let leaf_spacing_b = add(
        &mut g,
        GeoNode::Math { op: MathOp::Mul, a: 40.0, b: 1.33 },
        at(7.0, 4.5),
    );
    g.edges.insert((leaf_spacing_b, 0), leaf_spacing);
    let leaf_pts_b = add(
        &mut g,
        GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 53.0, count: 32 },
        at(8.0, 4.0),
    );
    g.edges.insert((leaf_pts_b, 0), branches);
    g.edges.insert((leaf_pts_b, 1), leaf_spacing_b);
    // The leaf: a disc off the origin, squashed into a blade pointing along
    // +x, so an Instance about the origin hangs it off the branch by its
    // base. Eight sides: at leaf size the corners are under a pixel, and
    // the point budget goes on more leaves instead.
    let leaf_disc = add(
        &mut g,
        GeoNode::Polygon {
            center: [0.022, 0.0],
            radius: 0.024,
            sides: 8,
            step: 1,
            rotation: 0.0,
            style: style([0.25, 0.62, 0.22, 1.0], dark_green, 1.5),
        },
        at(7.0, 6.0),
    );
    let leaf = add(
        &mut g,
        GeoNode::Transform { translate: [0.0, 0.0], rotate: 0.0, scale: [1.0, 0.42], pivot: [0.0, 0.0], inherit: false },
        at(8.0, 6.0),
    );
    g.edges.insert((leaf, 0), leaf_disc);
    let leaves_a = add(
        &mut g,
        GeoNode::Instance { scale: 0.6, rotation: 0.9, inherit_cd: true, pivot: Pivot::Origin },
        at(9.0, 3.0),
    );
    g.edges.insert((leaves_a, 0), leaf);
    g.edges.insert((leaves_a, 1), leaf_pts_a);
    g.edges.insert((leaves_a, 2), leaf_size);
    let leaves_b = add(
        &mut g,
        GeoNode::Instance { scale: 0.6, rotation: -0.9, inherit_cd: true, pivot: Pivot::Origin },
        at(9.0, 4.0),
    );
    g.edges.insert((leaves_b, 0), leaf);
    g.edges.insert((leaves_b, 1), leaf_pts_b);
    g.edges.insert((leaves_b, 2), leaf_size);

    // The flower: a star at the stem's tip, its centre disc, both swaying
    // with the same wind so they stay on the stem.
    let inner = add(&mut g, GeoNode::Math { op: MathOp::Mul, a: 0.075, b: 0.5 }, at(1.0, 5.0));
    g.edges.insert((inner, 0), bloom);
    let flower = add(
        &mut g,
        GeoNode::Star {
            center: [0.5, 0.20],
            outer: 0.075,
            inner: 0.037,
            points: 8,
            rotation: 0.0,
            style: style([0.95, 0.45, 0.25, 1.0], [0.6, 0.2, 0.1, 1.0], 2.0),
        },
        at(2.0, 0.0),
    );
    g.edges.insert((flower, 2), bloom);
    g.edges.insert((flower, 3), inner);
    let flower_sway = add(
        &mut g,
        GeoNode::Displace { mode: WarpMode::Offset, amount: 0.03, x: 0.5, y: 0.5 },
        at(3.0, 0.0),
    );
    g.edges.insert((flower_sway, 0), flower);
    g.edges.insert((flower_sway, 1), wind_x);
    g.edges.insert((flower_sway, 2), wind_y);
    g.edges.insert((flower_sway, 3), wind);
    let disc_r = add(&mut g, GeoNode::Math { op: MathOp::Mul, a: 0.075, b: 0.35 }, at(1.0, 6.0));
    g.edges.insert((disc_r, 0), bloom);
    let disc = add(
        &mut g,
        GeoNode::Polygon {
            center: [0.5, 0.20],
            radius: 0.026,
            sides: 24,
            step: 1,
            rotation: 0.0,
            style: style([1.0, 0.85, 0.3, 1.0], [0.7, 0.5, 0.1, 1.0], 1.5),
        },
        at(2.0, 1.0),
    );
    g.edges.insert((disc, 2), disc_r);
    let disc_sway = add(
        &mut g,
        GeoNode::Displace { mode: WarpMode::Offset, amount: 0.03, x: 0.5, y: 0.5 },
        at(3.0, 1.0),
    );
    g.edges.insert((disc_sway, 0), disc);
    g.edges.insert((disc_sway, 1), wind_x);
    g.edges.insert((disc_sway, 2), wind_y);
    g.edges.insert((disc_sway, 3), wind);

    // Back to front: ground, stem, leaves, flower, its centre.
    let mut below = ground;
    for (k, layer) in
        [stem_style, branches, leaves_a, leaves_b, flower_sway, disc_sway].into_iter().enumerate()
    {
        let m = add(&mut g, GeoNode::Merge { count: 2 }, at(10.0 + k as f32, 5.0));
        g.edges.insert((m, 0), below);
        g.edges.insert((m, 1), layer);
        below = m;
    }
    let out = add(&mut g, GeoNode::Output, at(16.0, 5.0));
    g.edges.insert((out, 0), below);
    g.next_id = id + 1;

    // --- the comp ------------------------------------------------------
    // Wind: a slow, broad fBm (the generator's own speed times the node's
    // cycles per frame), so the whole plant leans together with a little
    // variation along the stem.
    let mut wind_field = FieldGraph::default();
    wind_field.nodes.insert(1, FieldNode::default());
    wind_field.layout.insert(1, [0.0, 0.0]);
    wind_field.next_id = 2;
    let mut comp = EditorGraph::default();
    let noise = comp.add(
        NodeKind::Noise(NoiseParams {
            size: (1920, 1080),
            graph: wind_field,
            scale: 1.0,
            seed: 5.0,
            speed: 0.02,
            ..Default::default()
        }),
        egui::pos2(60.0, 120.0),
    );
    let plant = comp.add(
        NodeKind::Geo(GeoParams { size: (1920, 1080), graph: g, seed: 3.0, inherit: false, ..Default::default() }),
        egui::pos2(320.0, 120.0),
    );
    comp.edges.push((plant, 0, noise));
    comp.view = Some(plant);
    comp
}

/// The "jelly title" 2dini preset as a whole setup, BAKED so it plays
/// straight away: one 2dini node, its preset wired to an Output. Re-bake
/// on the Physics node after changing anything upstream.
pub fn jelly_title() -> EditorGraph {
    let mut p = jelly_geo();
    // The setup's script runs 1-200 (`JELLY_RANGE`): the sim starts on its
    // first frame, so all 200 frames of it play.
    for n in p.graph.nodes.values_mut() {
        if let GeoNode::Physics { settings, .. } = n {
            settings.start = JELLY_RANGE.0 as i64;
        }
    }
    let bake = bake_of(&p.graph);
    if let Some(slot) = physics_slot(&mut p) {
        *slot = engine::geophys::BakeSlot(Some(std::sync::Arc::new(bake)));
    }
    let mut comp = EditorGraph::default();
    let jelly = comp.add(NodeKind::Geo(p), egui::pos2(200.0, 120.0));
    comp.view = Some(jelly);
    comp
}

/// The context the jelly title is baked in: the scatter seed and the
/// format the layer draws with (a bake is only good for those).
const JELLY_SEED: f32 = 3.0;
const JELLY_SIZE: (u32, u32) = (1920, 1080);
const JELLY_FPS: f32 = 24.0;
/// The jelly title setup's script range (the web build opens on it).
pub const JELLY_RANGE: (u64, u64) = (1, 200);

/// The jelly title's 2dini, UNBAKED: the preset wired to an Output.
fn jelly_geo() -> GeoParams {
    let preset = engine::geopresets::PRESETS.iter().find(|p| p.name == "jelly title").unwrap();
    let mut g = (preset.build)();
    let phys = g.out.unwrap();
    let out = g.next_id;
    g.next_id += 1;
    g.nodes.insert(out, GeoNode::Output);
    let [x, y] = g.layout.get(&phys).copied().unwrap_or([0.0, 0.0]);
    g.layout.insert(out, [x + 190.0, y]);
    g.edges.insert((out, 0), phys);
    g.out = Some(out);
    GeoParams { size: JELLY_SIZE, graph: g, seed: JELLY_SEED, inherit: false, ..Default::default() }
}

fn physics_slot(p: &mut GeoParams) -> Option<&mut engine::geophys::BakeSlot> {
    p.graph.nodes.values_mut().find_map(|n| match n {
        GeoNode::Physics { bake, .. } => Some(bake),
        _ => None,
    })
}

/// Simulate the jelly title from its start frame, with the context the
/// layer draws with. Seconds of CPU: never on the UI thread of a running
/// app (see `warm_default_geo`).
fn jelly_bake() -> engine::geophys::PhysicsBake {
    bake_of(&jelly_geo().graph)
}

fn bake_of(g: &engine::geo::GeoGraph) -> engine::geophys::PhysicsBake {
    let (phys, settings) = g
        .nodes
        .iter()
        .find_map(|(id, n)| match n {
            GeoNode::Physics { settings, .. } => Some((*id, settings.clone())),
            _ => None,
        })
        .unwrap();
    let (w, h) = JELLY_SIZE;
    let (aspect, ppu) = (h as f32 / w as f32, w as f32);
    let cx = engine::geo::EvalCx { field: None, tick: settings.start as f32, seed: JELLY_SEED, aspect, px_per_unit: ppu };
    let bodies = g.input_of(phys, 0).map(|s| engine::geo::eval_node(g, s, &cx)).unwrap_or_default();
    engine::geophys::bake(&bodies, &Default::default(), &settings, aspect, ppu, JELLY_FPS)
}

/// The jelly title's bake, once `warm_default_geo` has made it.
static JELLY_BAKE: std::sync::OnceLock<std::sync::Arc<engine::geophys::PhysicsBake>> =
    std::sync::OnceLock::new();

/// Bake the default 2dini's sim on a worker, so a 2dini added later plays
/// at once instead of freezing the UI for the seconds a bake takes.
/// Called once at startup by an interactive run.
pub fn warm_default_geo() {
    crate::bg::spawn(|| {
        let b = std::sync::Arc::new(jelly_bake());
        let _ = JELLY_BAKE.set(b);
    });
}

/// What a NEW 2dini starts as: the jelly title, the word wobble as a
/// soft-body sim — baked when the startup warm-up has finished (it takes a
/// few seconds), else unbaked (the blobs stand still until the Physics
/// node inside is baked). Only the add menus use this: a loaded setup's
/// 2dini is whatever it saved, and `GeoParams::default()` stays empty.
pub fn default_geo() -> GeoParams {
    let mut p = jelly_geo();
    if let (Some(b), Some(slot)) = (JELLY_BAKE.get(), physics_slot(&mut p)) {
        *slot = engine::geophys::BakeSlot(Some(b.clone()));
    }
    p
}

#[cfg(test)]
mod default_geo_tests {
    use super::*;

    /// A new 2dini is the jelly title wired to an Output, and it takes the
    /// warm bake once there is one; a LOADED 2dini is untouched, because
    /// `GeoParams::default()` (what serde fills a missing field from) is
    /// still the empty graph.
    #[test]
    fn a_new_2dini_is_the_soft_body_wobble() {
        assert!(GeoParams::default().graph.nodes.is_empty());
        let mut p = default_geo();
        let out = p.graph.out.unwrap();
        assert_eq!(p.graph.nodes.get(&out), Some(&GeoNode::Output));
        let phys = *p.graph.edges.get(&(out, 0)).unwrap();
        assert!(matches!(p.graph.nodes.get(&phys), Some(GeoNode::Physics { .. })));
        assert!(p.graph.nodes.values().any(|n| matches!(n, GeoNode::SoftBody { .. })));
        assert!(p.graph.nodes.values().any(|n| matches!(n, GeoNode::Text { text, .. } if text == "wobble")));
        assert!(physics_slot(&mut p).unwrap().0.is_none(), "nothing warmed yet");
        let fake = engine::geophys::PhysicsBake {
            start: 0,
            frames: 1,
            fps: JELLY_FPS,
            stride: 0,
            bodies: Vec::new(),
            data: Vec::new(),
            input_hash: 0,
            skipped: 0,
        };
        let _ = JELLY_BAKE.set(std::sync::Arc::new(fake));
        let mut p = default_geo();
        assert!(physics_slot(&mut p).unwrap().0.is_some(), "the warm bake is used");
    }
}
