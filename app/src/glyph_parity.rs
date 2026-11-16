//! The colour designer (`colors/`) draws Wobble's node icons from its own
//! copy, `colors/src/glyphs.rs` — it cannot link the app. This includes that
//! copy, paints every node both ways and fails on any difference, naming the
//! node. Changed a glyph? Copy the change into `colors/src/glyphs.rs`.

#[path = "../../colors/src/glyphs.rs"]
mod designer;

use crate::editor::{node_catalog, paint_bar, paint_body, type_color};
use crate::theme;
use designer::{Icon, NODES};
use egui::epaint::ClippedShape;
use egui::{Color32, Pos2, Stroke};
use engine::graph::NodeKind;

/// The designer icon a node wears; `None` for a Dot, which wears its wire's
/// colour and is not in the designer.
fn icon_of(kind: &NodeKind) -> Option<Icon> {
    Some(match kind {
        NodeKind::Read { .. } => Icon::Read,
        NodeKind::Grid(_) => Icon::Grid,
        NodeKind::Gradient(_) => Icon::Gradient,
        NodeKind::Blank(_) => Icon::Blank,
        NodeKind::Starfield(_) => Icon::Starfield,
        NodeKind::Splat(_) => Icon::Splat,
        NodeKind::Planet(_) => Icon::Planet,
        NodeKind::Plasma(_) => Icon::Plasma,
        NodeKind::Particles(_) | NodeKind::ParticleNode(_) => Icon::Particles,
        NodeKind::GlintKernel(_) => Icon::GlintKernel,
        NodeKind::BokehKernel(_) => Icon::BokehKernel,
        NodeKind::Grain(_) => Icon::Grain,
        NodeKind::Noise(_)
        | NodeKind::Field(_)
        | NodeKind::GeoNode(engine::geo::GeoNode::Noise { .. }) => Icon::Noise,
        NodeKind::Geo(_) | NodeKind::GeoNode(_) => Icon::Geo,
        NodeKind::Sdf(_) | NodeKind::SdfNode(_) => Icon::Sdf,
        NodeKind::Blender(_) => Icon::Blender,
        NodeKind::Paint(_) => Icon::Paint,
        NodeKind::Grade(_) => Icon::Grade,
        NodeKind::Mult(_) => Icon::Mult,
        NodeKind::Curve(_) => Icon::Curve,
        NodeKind::Saturation(_) => Icon::Saturation,
        NodeKind::Contrast(_) => Icon::Contrast,
        NodeKind::ColorCompress(_) => Icon::ColorCompress,
        NodeKind::HueRotate(_) => Icon::HueRotate,
        NodeKind::Lut(_) => Icon::Lut,
        NodeKind::Text(_) => Icon::Text,
        NodeKind::Burnin(_) => Icon::Burnin,
        NodeKind::Mirror(_) => Icon::Mirror,
        NodeKind::LensDistort(_) => Icon::LensDistort,
        NodeKind::FlipFlop(_) => Icon::FlipFlop,
        NodeKind::Turn(_) => Icon::Turn,
        NodeKind::Gamma(_) => Icon::Gamma,
        NodeKind::ToneMap(_) => Icon::ToneMap,
        NodeKind::Expression(_) => Icon::Expression,
        NodeKind::Despill(_) => Icon::Despill,
        NodeKind::Lightwrap(_) => Icon::Lightwrap,
        NodeKind::Upscale(_) => Icon::Upscale,
        NodeKind::HdrCnn(_) => Icon::HdrCnn,
        NodeKind::SeedVr(_) => Icon::SeedVr,
        NodeKind::Depth(_) => Icon::Depth,
        NodeKind::LtxHdr(_) => Icon::LtxHdr,
        NodeKind::ScreenKey(_) => Icon::ScreenKey,
        NodeKind::CorridorKey(_) => Icon::CorridorKey,
        NodeKind::Sammie(_) => Icon::Sammie,
        NodeKind::ColorSlice(_) => Icon::Tinter,
        NodeKind::Blur(_) => Icon::Blur,
        NodeKind::MotionBlur(_) => Icon::MotionBlur,
        NodeKind::Flow(_) => Icon::Flow,
        NodeKind::Rifetime(_) => Icon::Rifetime,
        NodeKind::Freeze(_) => Icon::Freeze,
        NodeKind::FrameFix(_) => Icon::FrameFix,
        NodeKind::VectorBlur(_) => Icon::VectorBlur,
        NodeKind::Camera3D(_) => Icon::Camera3d,
        NodeKind::Card3D(_) => Icon::Card3d,
        NodeKind::Scene3D(_) => Icon::Scene3d,
        NodeKind::Render3D(_) => Icon::Render3d,
        NodeKind::VectorWarp(_) => Icon::VectorWarp,
        NodeKind::Defocus(_) => Icon::Defocus,
        NodeKind::Glow(_) => Icon::Glow,
        NodeKind::Godrays(_) => Icon::Godrays,
        NodeKind::LensFlare(_) => Icon::LensFlare,
        NodeKind::EdgeExtend(_) => Icon::EdgeExtend,
        NodeKind::Glint(_) => Icon::Glint,
        NodeKind::EdgeDetect(_) => Icon::EdgeDetect,
        NodeKind::Median(_) => Icon::Median,
        NodeKind::HighlightRebuild(_) => Icon::HighlightRebuild,
        NodeKind::Stabilize(_) => Icon::Stabilize,
        NodeKind::CamTrack(_) => Icon::CameraTrack,
        NodeKind::Convolve(_) => Icon::Convolve,
        NodeKind::Diffuse(_) => Icon::Diffuse,
        NodeKind::Dilate(_) => Icon::Dilate,
        NodeKind::Transform(_) => Icon::Transform,
        NodeKind::Reformat(_) => Icon::Reformat,
        NodeKind::CornerPin(_) => Icon::CornerPin,
        NodeKind::SoftCrop(_) => Icon::SoftCrop,
        NodeKind::Spline(_) => Icon::Spline,
        NodeKind::LumaKey(_) => Icon::LumaKey,
        NodeKind::Keyer(_) => Icon::Keyer,
        NodeKind::Cryptomatte(_) => Icon::Cryptomatte,
        NodeKind::MatteCombine(_) => Icon::MatteCombine,
        NodeKind::Mask => Icon::Mask,
        NodeKind::Extract(_) => Icon::Extract,
        NodeKind::MatteIn => Icon::MatteIn,
        NodeKind::ToMatte(_) => Icon::ToMatte,
        NodeKind::ToRgba(_) => Icon::ToRgba,
        NodeKind::ToData(_) => Icon::ToData,
        NodeKind::UvWarp(_) => Icon::UvWarp,
        NodeKind::Env(_) => Icon::Env,
        NodeKind::Refract(_) => Icon::Refract,
        NodeKind::GridWarp(_) => Icon::GridWarp,
        NodeKind::Warp(_) => Icon::Warp,
        NodeKind::Premult { premult: true } => Icon::Premult,
        NodeKind::Premult { premult: false } => Icon::Unpremult,
        NodeKind::Breakout(_) => Icon::Breakout,
        NodeKind::Recombine(_) => Icon::Recombine,
        NodeKind::ChannelSelect(_) => Icon::ChannelSelect,
        NodeKind::Shuffle(_) => Icon::Shuffle,
        NodeKind::Blend { .. } => Icon::Blend,
        NodeKind::Mix(_) => Icon::Mix,
        NodeKind::Layer(_) => Icon::Layer,
        NodeKind::Switch(_) => Icon::Switch,
        NodeKind::Slip(_) => Icon::Slip,
        NodeKind::Speed(_) => Icon::Speed,
        NodeKind::TimeRemap(_) => Icon::TimeRemap,
        NodeKind::RangeRemap(_) => Icon::RangeRemap,
        NodeKind::TimeRange(_) => Icon::TimeRange,
        NodeKind::LoopIn => Icon::LoopIn,
        NodeKind::LoopOut(_) => Icon::LoopOut,
        NodeKind::GraphBake { .. } => Icon::Bake,
        NodeKind::Tracker(_) => Icon::Tracker,
        NodeKind::PlanarTrack(_) => Icon::PlanarTrack,
        NodeKind::Write(_) => Icon::Write,
        NodeKind::Shader(_) => Icon::Shader,
        NodeKind::Custom { .. } => Icon::Plugin,
        NodeKind::Prompt(_) => Icon::Prompt,
        NodeKind::Dot { .. } | NodeKind::GenAi(_) | NodeKind::GenAiStep(_) | NodeKind::RotoNode(_) => return None,
    })
}

/// What a painter received, in order.
fn shapes(paint: impl FnOnce(&egui::Painter)) -> Vec<ClippedShape> {
    let ctx = egui::Context::default();
    let mut paint = Some(paint);
    ctx.run_ui(egui::RawInput::default(), |ui| {
        if let Some(f) = paint.take() {
            f(&ui.ctx().layer_painter(egui::LayerId::background()));
        }
    })
    .shapes
}

/// The graph's node at `scale` (the way `show()` paints it) against the
/// designer's.
fn both(kind: &NodeKind, icon: Icon, scale: f32) -> (Vec<ClippedShape>, Vec<ClippedShape>) {
    let sp = Pos2::new(300.0, 200.0);
    let fill = Color32::from_rgb(0x8a, 0x4f, 0x9e);
    let stroke = Stroke::new(1.0, Color32::from_gray(12));
    let r_node = designer::R * scale;
    let graph = shapes(|p| {
        let r = paint_bar(p, kind, sp, scale, fill, stroke).unwrap_or(r_node);
        paint_body(p, kind, sp, r, scale, fill, stroke);
    });
    let copy = shapes(|p| {
        designer::paint(p, icon, sp, scale, fill, stroke);
    });
    (graph, copy)
}

#[test]
fn the_designer_draws_every_node_the_way_the_graph_does() {
    let mut kinds: Vec<(String, NodeKind)> =
        node_catalog().iter().map(|e| (e.label.to_string(), (e.make)(0))).collect();
    kinds.push((
        "a plugin".into(),
        NodeKind::Custom { type_id: "no.such.plugin".into(), params: Default::default() },
    ));
    let mut stale = Vec::new();
    for (label, kind) in &kinds {
        let Some(icon) = icon_of(kind) else { continue };
        for scale in [1.0, 0.45] {
            let (graph, copy) = both(kind, icon, scale);
            assert!(!graph.is_empty(), "{label}: the graph painted nothing");
            if graph != copy {
                stale.push(format!("{label} ({icon:?}) at zoom {scale}"));
                break;
            }
        }
    }
    assert!(
        stale.is_empty(),
        "the colour designer's icons no longer match the graph — copy the \
         changed glyphs into colors/src/glyphs.rs: {stale:?}"
    );
}

#[test]
fn every_node_in_the_add_menus_is_in_the_designer_in_its_own_colour() {
    for e in node_catalog() {
        let kind = (e.make)(0);
        let Some(icon) = icon_of(&kind) else { continue };
        let node = NODES.iter().find(|n| n.icon == icon).unwrap_or_else(|| {
            panic!("{}: no entry in colors/src/glyphs.rs NODES", e.label)
        });
        assert_eq!(
            theme::get(node.slot),
            type_color(&kind),
            "{}: the designer recolours {:?}, the graph paints it in another slot",
            e.label,
            node.slot
        );
    }
}
