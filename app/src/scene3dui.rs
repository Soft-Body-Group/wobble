//! The 3D cards nodes' app glue beyond their panels: the demo setup
//! (`examples/cards3d.wbl`), and — as they land — file import and the
//! Blender sync. The data and the lowering are `engine::scene3d`.

use crate::editor::EditorGraph;
use engine::graph::{CardParams, NodeKind, SceneParams};
use engine::scene3d::import::TrackKind;
use engine::scene3d::Source;

/// A Card for every transform `source` (a file, or a Blender node's
/// scene) holds, each reading its object from the same source, gathered
/// into a new Blend 3D, laid out below the camera that asked.
pub fn cards_from(g: &mut EditorGraph, camera: u64, source: &Source) {
    let path = match source {
        Source::File { path, .. } => path.clone(),
        Source::Blender { node, .. } => match engine::scene3d::blender_scene(*node) {
            Some((p, _)) => p,
            None => return,
        },
        Source::Wobble | Source::Tracker { .. } => return,
    };
    let Ok(file) = engine::scene3d::import::load(&path) else { return };
    let base = g.node(camera).map(|n| n.pos).unwrap_or_default();
    let names: Vec<String> =
        file.tracks.iter().filter(|t| t.kind == TrackKind::Xform).map(|t| t.name.clone()).collect();
    if names.is_empty() {
        return;
    }
    let n = names.len().min(engine::scene3d::MAX_SCENE_INPUTS);
    let scene = g.add(NodeKind::Scene3D(SceneParams { inputs: n }), base + egui::vec2(0.0, 220.0));
    for (i, name) in names.into_iter().take(n).enumerate() {
        let source = match source {
            Source::File { path, offset, .. } => Source::File { path: path.clone(), object: name, offset: *offset },
            Source::Blender { node, .. } => Source::Blender { node: *node, object: name },
            Source::Wobble | Source::Tracker { .. } => unreachable!(),
        };
        let card = g.add(
            NodeKind::Card3D(CardParams { source, ..Default::default() }),
            base + egui::vec2(-60.0 * (n as f32 - 1.0) / 2.0 + 60.0 * i as f32 - 120.0, 110.0),
        );
        g.connect(scene, i, card);
    }
}

#[cfg(test)]
mod tests {
    use crate::editor::EditorGraph;
    use egui::pos2;
    use engine::graph::{
        CameraParams, CardParams, GradStop, GradientParams, GridParams, NodeKind, Render3dParams, SceneParams,
    };
    use engine::scene3d::Xform;

    /// A room of cards with a camera dollying through it: a gridded floor,
    /// a gradient back wall, two side walls, and a small card standing in
    /// the middle that the camera passes. Motion blur on.
    pub fn demo() -> EditorGraph {
        let mut g = EditorGraph::default();
        let grid = g.add(
            NodeKind::Grid(GridParams {
                size: (1024, 1024),
                spacing: [64.0, 64.0],
                width: 3.0,
                line: [0.9, 0.9, 0.95, 1.0],
                back: [0.08, 0.09, 0.12, 1.0],
                ..Default::default()
            }),
            pos2(0.0, 0.0),
        );
        let wall = g.add(
            NodeKind::Gradient(GradientParams {
                size: (1920, 1080),
                stops: vec![
                    GradStop { pos: 0.0, color: [0.05, 0.12, 0.35, 1.0] },
                    GradStop { pos: 1.0, color: [0.65, 0.75, 0.95, 1.0] },
                ],
                angle: 90.0,
                ..Default::default()
            }),
            pos2(160.0, 0.0),
        );
        let sign = g.add(
            NodeKind::Grid(GridParams {
                size: (512, 512),
                spacing: [128.0, 128.0],
                width: 24.0,
                line: [0.95, 0.25, 0.3, 1.0],
                back: [0.0, 0.0, 0.0, 0.0],
                ..Default::default()
            }),
            pos2(320.0, 0.0),
        );
        let card = |rot: [f32; 3], at: [f32; 3], s: f32| CardParams {
            xform: Xform { translate: at, rotate: rot, scale: [s; 3] },
            ..Default::default()
        };
        let floor = g.add(NodeKind::Card3D(card([-90.0, 0.0, 0.0], [0.0, -1.0, -3.0], 6.0)), pos2(0.0, 120.0));
        let back = g.add(NodeKind::Card3D(card([0.0, 0.0, 0.0], [0.0, 0.69, -6.0], 6.0)), pos2(120.0, 120.0));
        let left = g.add(NodeKind::Card3D(card([0.0, 90.0, 0.0], [-3.0, 2.0, -3.0], 6.0)), pos2(240.0, 120.0));
        let right = g.add(NodeKind::Card3D(card([0.0, -90.0, 0.0], [3.0, 2.0, -3.0], 6.0)), pos2(360.0, 120.0));
        let mid = g.add(NodeKind::Card3D(card([0.0, 25.0, 0.0], [0.4, -0.3, -2.5], 1.2)), pos2(480.0, 120.0));
        g.connect(floor, 0, grid);
        g.connect(back, 0, wall);
        g.connect(left, 0, grid);
        g.connect(right, 0, grid);
        g.connect(mid, 0, sign);
        let scene = g.add(NodeKind::Scene3D(SceneParams { inputs: 5 }), pos2(240.0, 240.0));
        for (i, c) in [floor, back, left, right, mid].into_iter().enumerate() {
            g.connect(scene, i, c);
        }
        let cam = g.add(
            NodeKind::Camera3D(CameraParams {
                xform: Xform { translate: [0.0, 0.0, 2.5], ..Default::default() },
                focal: 28.0,
                ..Default::default()
            }),
            pos2(60.0, 240.0),
        );
        if let Some(n) = g.node_mut(cam) {
            use crate::anim::Channel;
            let mut key = |ch: Channel, f: u64, v: f32| n.anim.entry(ch).or_default().set_key(f, v);
            key(Channel::Cam3dTz, 0, 2.5);
            key(Channel::Cam3dTz, 48, -1.0);
            key(Channel::Cam3dTx, 0, -0.6);
            key(Channel::Cam3dTx, 48, 0.8);
            key(Channel::Cam3dRy, 0, -8.0);
            key(Channel::Cam3dRy, 48, 14.0);
        }
        let render = g.add(
            NodeKind::Render3D(Render3dParams {
                inherit: false,
                samples: 8,
                shutter: 0.5,
                ..Default::default()
            }),
            pos2(160.0, 360.0),
        );
        g.connect(render, 0, cam);
        g.connect(render, 1, scene);
        g.view = Some(render);
        g
    }


    /// `examples/cards3d.wbl` is regenerated here, so it can never drift
    /// from the nodes, and it has to load back through the app's own path
    /// and lower to a stage with every card.
    #[test]
    fn the_demo_is_written_and_lowers() {
        let g = demo();
        let empty_dirs: Vec<std::path::PathBuf> = Vec::new();
        let huds: Vec<crate::widgets::HudSlider> = Vec::new();
        let activated = vec![g.view.unwrap()];
        let pins: Vec<u64> = Vec::new();
        let layouts: Vec<crate::PropLayout> = Vec::new();
        let s = crate::SetupSave {
            version: 2,
            fps: 24.0,
            pinned_dirs: &empty_dirs,
            graph: std::borrow::Cow::Borrowed(&g),
            canvas: None,
            range: Some((0, 48)),
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
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/cards3d.wbl");
        if std::fs::read_to_string(&path).ok().as_deref() != Some(json.as_str()) {
            std::fs::write(&path, &json).unwrap();
        }
        let mut val: serde_json::Value = serde_json::from_str(&json).unwrap();
        crate::migrate_setup_json(&mut val);
        let l: crate::SetupLoad = serde_json::from_value(val).unwrap();
        let eg = l.graph.to_engine_graph_at_f(Some(24.0));
        let render = g.view.unwrap();
        let Some(NodeKind::Render3D(p)) = eg.nodes.get(&render) else { panic!("render gone") };
        let low = p.lowered.as_ref().expect("lowered");
        assert_eq!(low.cards.len(), 5);
        assert_eq!(low.cams.len(), 8, "a pose per shutter sample");
        let z: Vec<f32> = low.cams.iter().map(|c| c.xform.translate[2]).collect();
        assert!(z.windows(2).all(|w| w[1] < w[0]), "the dolly moves across the shutter: {z:?}");
        for i in 1..=5 {
            assert!(eg.input_of(render, i).is_some(), "card picture {i} wired");
        }
    }

    /// A camera read from a .chan file takes the file's pose at the frame
    /// (and across the shutter), and "cards for the rest" builds a Card per
    /// transform of a JSON file, each reading its object.
    #[test]
    fn file_sources_lower_and_make_cards() {
        let dir = std::env::temp_dir().join(format!("wobble-scene3d-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let chan = dir.join("cam.chan");
        std::fs::write(&chan, "0 0 0 10 0 0 0\n10 5 0 10 0 0 0\n").unwrap();
        let mut g = demo();
        let render = g.view.unwrap();
        let cam = g.input_of(render, 0).unwrap();
        if let Some(n) = g.node_mut(cam) {
            n.anim.clear();
            if let NodeKind::Camera3D(p) = &mut n.kind {
                p.source = engine::scene3d::Source::File {
                    path: chan.to_string_lossy().into_owned(),
                    object: String::new(),
                    offset: 0.0,
                };
            }
        }
        let eg = g.to_engine_graph_at_f(Some(4.0));
        let Some(NodeKind::Render3D(p)) = eg.nodes.get(&render) else { panic!() };
        let xs: Vec<f32> = p.lowered.as_ref().unwrap().cams.iter().map(|c| c.xform.translate[0]).collect();
        let mid = xs.iter().sum::<f32>() / xs.len() as f32;
        assert!((mid - 2.0).abs() < 1e-3, "frame 4 of a 0..10 move to x=5: {xs:?}");
        assert!(xs.windows(2).all(|w| w[1] > w[0]), "the shutter samples the file: {xs:?}");

        let json = dir.join("scene.json");
        std::fs::write(
            &json,
            r#"{ "objects": [ { "name": "cam", "type": "camera", "frames": [ { "frame": 0 } ] },
                             { "name": "a", "frames": [ { "frame": 0, "translate": [1, 0, 0] } ] },
                             { "name": "b", "frames": [ { "frame": 0, "translate": [2, 0, 0] } ] } ] }"#,
        )
        .unwrap();
        let before = g.nodes.len();
        let src = engine::scene3d::Source::File { path: json.to_string_lossy().into_owned(), object: String::new(), offset: 0.0 };
        super::cards_from(&mut g, cam, &src);
        assert_eq!(g.nodes.len(), before + 3, "a Scene and two Cards");
        let objects: Vec<String> = g
            .nodes
            .values()
            .filter_map(|n| match &n.kind {
                NodeKind::Card3D(CardParams { source: engine::scene3d::Source::File { object, .. }, .. }) => {
                    Some(object.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(objects.len(), 2);
        assert!(objects.contains(&"a".to_string()) && objects.contains(&"b".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
