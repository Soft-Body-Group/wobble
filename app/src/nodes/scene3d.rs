//! The 3D cards nodes' app half — Camera, Card, Scene and Render3D: their
//! keyable channels, panels and glyphs. The data and the lowering are
//! `engine::scene3d`; the GPU pass is `engine::nodes::render3d`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{CameraParams, CardParams, Render3dParams, SceneParams};
use engine::scene3d::{Source, Xform, MAX_SAMPLES, MAX_SCENE_INPUTS};

pub const CAMERA_CHANNELS: &[Channel] = &[
    Channel::Cam3dTx,
    Channel::Cam3dTy,
    Channel::Cam3dTz,
    Channel::Cam3dRx,
    Channel::Cam3dRy,
    Channel::Cam3dRz,
    Channel::Cam3dFocal,
    Channel::Cam3dApW,
    Channel::Cam3dApH,
    Channel::Cam3dNear,
    Channel::Cam3dFar,
];

pub const CARD_CHANNELS: &[Channel] = &[
    Channel::Card3dTx,
    Channel::Card3dTy,
    Channel::Card3dTz,
    Channel::Card3dRx,
    Channel::Card3dRy,
    Channel::Card3dRz,
    Channel::Card3dSx,
    Channel::Card3dSy,
    Channel::Card3dSz,
    Channel::Card3dAspect,
    Channel::Card3dOpacity,
];

/// Every channel the four nodes declare.
pub const CHANNELS: &[Channel] = &[
    Channel::Cam3dTx,
    Channel::Cam3dTy,
    Channel::Cam3dTz,
    Channel::Cam3dRx,
    Channel::Cam3dRy,
    Channel::Cam3dRz,
    Channel::Cam3dFocal,
    Channel::Cam3dApW,
    Channel::Cam3dApH,
    Channel::Cam3dNear,
    Channel::Cam3dFar,
    Channel::Card3dTx,
    Channel::Card3dTy,
    Channel::Card3dTz,
    Channel::Card3dRx,
    Channel::Card3dRy,
    Channel::Card3dRz,
    Channel::Card3dSx,
    Channel::Card3dSy,
    Channel::Card3dSz,
    Channel::Card3dAspect,
    Channel::Card3dOpacity,
    Channel::Render3dShutter,
    Channel::Render3dOffset,
];

pub const CAMERA_TILES: usize = 2;
pub const CARD_TILES: usize = 2;
pub const SCENE_TILES: usize = 1;
pub const RENDER_TILES: usize = 2;

fn xf_get(x: &Xform, i: usize) -> f32 {
    match i {
        0..=2 => x.translate[i],
        3..=5 => x.rotate[i - 3],
        _ => x.scale[i - 6],
    }
}

fn xf_set(x: &mut Xform, i: usize, v: f32) {
    match i {
        0..=2 => x.translate[i] = v,
        3..=5 => x.rotate[i - 3] = v,
        _ => x.scale[i - 6] = v,
    }
}

pub fn camera_get(p: &CameraParams, ch: Channel) -> Option<f32> {
    let i = CAMERA_CHANNELS.iter().position(|c| *c == ch)?;
    Some(match i {
        0..=5 => xf_get(&p.xform, i),
        6 => p.focal,
        7 => p.aperture[0],
        8 => p.aperture[1],
        9 => p.near,
        _ => p.far,
    })
}

pub fn camera_set(p: &mut CameraParams, ch: Channel, v: f32) {
    let Some(i) = CAMERA_CHANNELS.iter().position(|c| *c == ch) else { return };
    match i {
        0..=5 => xf_set(&mut p.xform, i, v),
        6 => p.focal = v,
        7 => p.aperture[0] = v,
        8 => p.aperture[1] = v,
        9 => p.near = v,
        _ => p.far = v,
    }
}

pub fn card_get(p: &CardParams, ch: Channel) -> Option<f32> {
    let i = CARD_CHANNELS.iter().position(|c| *c == ch)?;
    Some(match i {
        0..=8 => xf_get(&p.xform, i),
        9 => p.aspect,
        _ => p.opacity,
    })
}

pub fn card_set(p: &mut CardParams, ch: Channel, v: f32) {
    let Some(i) = CARD_CHANNELS.iter().position(|c| *c == ch) else { return };
    match i {
        0..=8 => xf_set(&mut p.xform, i, v),
        9 => p.aspect = v,
        _ => p.opacity = v,
    }
}

pub fn render_get(p: &Render3dParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Render3dShutter => Some(p.shutter),
        Channel::Render3dOffset => Some(p.offset),
        _ => None,
    }
}

pub fn render_set(p: &mut Render3dParams, ch: Channel, v: f32) {
    match ch {
        Channel::Render3dShutter => p.shutter = v,
        Channel::Render3dOffset => p.offset = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    Some(match ch {
        Channel::Cam3dTx => "camera translate x",
        Channel::Cam3dTy => "camera translate y",
        Channel::Cam3dTz => "camera translate z",
        Channel::Cam3dRx => "camera rotate x",
        Channel::Cam3dRy => "camera rotate y",
        Channel::Cam3dRz => "camera rotate z",
        Channel::Cam3dFocal => "camera focal length",
        Channel::Cam3dApW => "camera filmback width",
        Channel::Cam3dApH => "camera filmback height",
        Channel::Cam3dNear => "camera near",
        Channel::Cam3dFar => "camera far",
        Channel::Card3dTx => "card translate x",
        Channel::Card3dTy => "card translate y",
        Channel::Card3dTz => "card translate z",
        Channel::Card3dRx => "card rotate x",
        Channel::Card3dRy => "card rotate y",
        Channel::Card3dRz => "card rotate z",
        Channel::Card3dSx => "card scale x",
        Channel::Card3dSy => "card scale y",
        Channel::Card3dSz => "card scale z",
        Channel::Card3dAspect => "card aspect",
        Channel::Card3dOpacity => "card opacity",
        Channel::Render3dShutter => "render 3d shutter",
        Channel::Render3dOffset => "render 3d shutter offset",
        _ => return None,
    })
}

fn xform_rows(ui: &mut egui::Ui, rcx: &mut RowCtx, x: &mut Xform, ch: &[Channel], scale: bool) -> Option<HudSlider> {
    let [tx, ty, tz] = &mut x.translate;
    let h1 = widgets::drags(ui, rcx, "translate", &mut [tx, ty, tz], 0.01, (-10.0, 10.0), &ch[0..3]);
    let [rx, ry, rz] = &mut x.rotate;
    let h2 = widgets::drags(ui, rcx, "rotate", &mut [rx, ry, rz], 0.5, (-180.0, 180.0), &ch[3..6]);
    let h3 = if scale {
        let [sx, sy, sz] = &mut x.scale;
        widgets::drags_linked(ui, rcx, "scale", &mut [sx, sy, sz], 0.01, (0.0, 4.0), &ch[6..9])
    } else {
        None
    };
    h1.or(h2).or(h3)
}

/// What a panel's source row asks the app to do.
pub enum SourceAct {
    /// Make a Card (in a new Scene) for every transform this source holds
    /// but the camera, each reading its object from the same source.
    CardsFrom(Source),
}

/// The source row: keyed here, a file, or a Blender node's live scene;
/// the object and (for a file) frame offset; what it read; and "bake to
/// keys" (sample the source over `range` into this node's own curves and
/// go back to keyed). `kind` is the node as authored, for the bake;
/// `blenders` the Blender nodes in the comp, by id and name. The camera
/// row offers cards for the rest.
fn source_rows(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    kind: engine::graph::NodeKind,
    range: (u64, u64),
    chans: &[Channel],
    blenders: &[(u64, String)],
    trackers: &[(u64, String)],
) -> (Option<Source>, Option<SourceAct>) {
    use engine::graph::NodeKind;
    use engine::scene3d::import::TrackKind;
    let node = rcx.node;
    let (src, want) = match &kind {
        NodeKind::Camera3D(p) => (p.source.clone(), TrackKind::Camera),
        NodeKind::Card3D(p) => (p.source.clone(), TrackKind::Xform),
        _ => return (None, None),
    };
    let mut new = src.clone();
    let mut act = None;
    let mut baked = false;
    ui.horizontal(|ui| {
        ui.label("source");
        let label = match &new {
            Source::Wobble => "keyed here",
            Source::File { .. } => "file",
            Source::Blender { .. } => "Blender",
            Source::Tracker { .. } => "tracker",
        };
        egui::ComboBox::from_id_salt(("scene3d-source", node)).selected_text(label).show_ui(ui, |ui| {
            if ui.selectable_label(new.is_wobble(), "keyed here").clicked() {
                new = Source::Wobble;
            }
            if ui.selectable_label(matches!(new, Source::File { .. }), "file").clicked() && !matches!(new, Source::File { .. }) {
                new = Source::File { path: String::new(), object: String::new(), offset: 0.0 };
            }
            let first = blenders.first().map(|b| b.0);
            if ui
                .add_enabled(first.is_some(), egui::Button::selectable(matches!(new, Source::Blender { .. }), "Blender"))
                .on_disabled_hover_text("add a Blender node first")
                .clicked()
                && !matches!(new, Source::Blender { .. })
            {
                new = Source::Blender { node: first.unwrap_or(0), object: String::new() };
            }
            if want == TrackKind::Camera {
                let first = trackers.first().map(|b| b.0);
                if ui
                    .add_enabled(first.is_some(), egui::Button::selectable(matches!(new, Source::Tracker { .. }), "tracker"))
                    .on_disabled_hover_text("add a CameraTrack node first")
                    .clicked()
                    && !matches!(new, Source::Tracker { .. })
                {
                    new = Source::Tracker { node: first.unwrap_or(0) };
                }
            }
        });
    });
    if let Source::Tracker { node: tn } = &mut new {
        ui.horizontal(|ui| {
            ui.label("CameraTrack");
            let shown = trackers.iter().find(|x| x.0 == *tn).map(|x| x.1.clone()).unwrap_or_else(|| "gone".into());
            egui::ComboBox::from_id_salt(("scene3d-tracker", node)).selected_text(shown).show_ui(ui, |ui| {
                for (id, name) in trackers {
                    if ui.selectable_label(*tn == *id, name).clicked() {
                        *tn = *id;
                    }
                }
            });
        });
        match engine::scene3d::tracker_solve(*tn) {
            Some(t) => {
                let s = &t.solve.solve;
                ui.weak(format!("{} frames solved, {} points, {:.2} px", s.poses.len(), s.points.len(), s.mean_error()));
            }
            None => {
                ui.colored_label(crate::theme::error(), "the CameraTrack has no solve yet");
            }
        }
        if ui
            .button("bake to keys")
            .on_hover_text("sample the solve over the script range into this node's own curves")
            .clicked()
        {
            baked = bake_to_keys(rcx, &kind, range, chans);
        }
        let out = if baked { Some(Source::Wobble) } else { (new != src).then_some(new) };
        return (out, None);
    }
    let file = match &mut new {
        Source::Wobble => return ((new != src).then_some(new), None),
        Source::Tracker { .. } => unreachable!(),
        Source::File { path, .. } => {
            ui.horizontal(|ui| {
                ui.label("file");
                let w = (ui.available_width() - 30.0).max(40.0);
                ui.add(egui::TextEdit::singleline(path).id(egui::Id::new(("scene3d-path", node))).desired_width(w).hint_text("json, chan or usda"));
                if ui.small_button("…").on_hover_text("pick a file").clicked() {
                    if let Some(f) = crate::filedialog::FileDialog::new()
                        .add_filter("cameras and transforms", &["json", "chan", "usda"])
                        .pick_file()
                    {
                        *path = f.to_string_lossy().into_owned();
                    }
                }
            });
            (!path.is_empty()).then(|| engine::scene3d::import::load(path))
        }
        Source::Blender { node: b, .. } => {
            ui.horizontal(|ui| {
                ui.label("Blender node");
                let shown = blenders.iter().find(|x| x.0 == *b).map(|x| x.1.clone()).unwrap_or_else(|| "gone".into());
                egui::ComboBox::from_id_salt(("scene3d-blender", node)).selected_text(shown).show_ui(ui, |ui| {
                    for (id, name) in blenders {
                        if ui.selectable_label(*b == *id, name).clicked() {
                            *b = *id;
                        }
                    }
                });
            });
            Some(match engine::scene3d::blender_scene(*b) {
                Some((path, _)) => engine::scene3d::import::load(&path),
                None => Err("waiting for the Blender scene (is the Blender node connected?)".into()),
            })
        }
    };
    let blender = matches!(new, Source::Blender { .. });
    let (object, offset) = match &mut new {
        Source::File { object, offset, .. } => (object, Some(offset)),
        Source::Blender { object, .. } => (object, None),
        Source::Wobble | Source::Tracker { .. } => unreachable!(),
    };
    match &file {
        Some(Ok(f)) => {
            let names: Vec<String> =
                f.tracks.iter().filter(|t| t.kind == want || want == TrackKind::Xform).map(|t| t.name.clone()).collect();
            ui.horizontal(|ui| {
                ui.label("object");
                let first = if blender && want == TrackKind::Camera { "scene camera" } else { "first" };
                let shown = if object.is_empty() { first.to_string() } else { object.clone() };
                egui::ComboBox::from_id_salt(("scene3d-object", node)).selected_text(shown).show_ui(ui, |ui| {
                    if ui.selectable_label(object.is_empty(), first).clicked() {
                        object.clear();
                    }
                    for n in &names {
                        if ui.selectable_label(object == n, n).clicked() {
                            *object = n.clone();
                        }
                    }
                });
            });
            if let Some(offset) = offset {
                widgets::plain_slider(ui, "frame offset", offset, -1000.0, 1000.0, true);
            }
            let frames: usize = f.track(object, want).map(|t| t.keys.len()).unwrap_or(0);
            let fps = f.fps.map(|r| format!(", {r} fps")).unwrap_or_default();
            ui.weak(format!(
                "{} object{}, {frames} key{}{fps}",
                f.tracks.len(),
                if f.tracks.len() == 1 { "" } else { "s" },
                if frames == 1 { "" } else { "s" }
            ));
        }
        Some(Err(e)) => {
            ui.colored_label(crate::theme::error(), e);
        }
        None => {}
    }
    ui.horizontal(|ui| {
        let ok = matches!(file, Some(Ok(_)));
        if ui
            .add_enabled(ok, egui::Button::new("bake to keys"))
            .on_hover_text("sample the source over the script range into this node's own curves")
            .clicked()
        {
            baked = bake_to_keys(rcx, &kind, range, chans);
        }
        if want == TrackKind::Camera
            && ui
                .add_enabled(ok, egui::Button::new("cards for the rest…"))
                .on_hover_text("a Card for every other object, gathered in a Blend 3D")
                .clicked()
        {
            act = Some(SourceAct::CardsFrom(new.clone()));
        }
    });
    // A bake goes back to keyed; anything else the row changed stands.
    let out = if baked { Some(Source::Wobble) } else { (new != src).then_some(new) };
    (out, act)
}

/// Sample a file-sourced node over `range` into its own curves.
fn bake_to_keys(rcx: &mut RowCtx, kind: &engine::graph::NodeKind, range: (u64, u64), chans: &[Channel]) -> bool {
    let mut cols: Vec<Vec<(u64, f32)>> = vec![Vec::new(); chans.len()];
    for f in range.0..=range.1 {
        let mut k = kind.clone();
        crate::anim::apply_at(&mut k, rcx.anim, f as f32);
        if engine::scene3d::apply_file_source(&mut k, f as f32).is_err() {
            return false;
        }
        for (c, ch) in chans.iter().enumerate() {
            if let Some(v) = crate::anim::get(&k, *ch) {
                cols[c].push((f, v));
            }
        }
    }
    for (ch, col) in chans.iter().zip(cols) {
        if crate::anim::is_locked(rcx.anim, *ch) {
            continue;
        }
        let curve = rcx.anim.entry(*ch).or_default();
        curve.keys.clear();
        for (f, v) in col {
            curve.set_key(f, v);
        }
    }
    rcx.anim.retain(|_, c| !c.keys.is_empty() || c.locked);
    true
}

pub fn camera_panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut CameraParams,
    range: (u64, u64),
    blenders: &[(u64, String)],
    trackers: &[(u64, String)],
) -> (Option<HudSlider>, Option<SourceAct>) {
    let (src, act) = source_rows(ui, rcx, engine::graph::NodeKind::Camera3D(p.clone()), range, CAMERA_CHANNELS, blenders, trackers);
    if let Some(s) = src {
        p.source = s;
    }
    if !p.source.is_wobble() {
        ui.weak("pose and optics come from the source");
        return (None, act);
    }
    let h = xform_rows(ui, rcx, &mut p.xform, CAMERA_CHANNELS, false);
    let h2 = widgets::slider(ui, rcx, "focal mm", &mut p.focal, 8.0, 300.0, &[Channel::Cam3dFocal]);
    let [aw, ah] = &mut p.aperture;
    let h3 = widgets::drags_linked(ui, rcx, "filmback mm", &mut [aw, ah], 0.1, (1.0, 70.0), &[Channel::Cam3dApW, Channel::Cam3dApH]);
    let h4 = widgets::drags(
        ui,
        rcx,
        "near / far",
        &mut [&mut p.near, &mut p.far],
        0.01,
        (0.001, 10000.0),
        &[Channel::Cam3dNear, Channel::Cam3dFar],
    );
    p.near = p.near.max(0.0);
    p.far = p.far.max(p.near + 1e-3);
    ui.weak("Y up, looking down -Z; the filmback width fits the render's width");
    (h.or(h2).or(h3).or(h4), act)
}

pub fn card_panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut CardParams,
    range: (u64, u64),
    blenders: &[(u64, String)],
) -> Option<HudSlider> {
    let (src, _) = source_rows(ui, rcx, engine::graph::NodeKind::Card3D(p.clone()), range, CARD_CHANNELS, blenders, &[]);
    if let Some(s) = src {
        p.source = s;
    }
    if !p.source.is_wobble() {
        ui.weak("the transform below rides on the object's");
    }
    let h = xform_rows(ui, rcx, &mut p.xform, CARD_CHANNELS, true);
    let h2 = widgets::slider(ui, rcx, "aspect", &mut p.aspect, 0.0, 4.0, &[Channel::Card3dAspect]);
    let h3 = widgets::slider(ui, rcx, "opacity", &mut p.opacity, 0.0, 1.0, &[Channel::Card3dOpacity]);
    ui.weak("one unit wide before scale; aspect 0 = the picture's own");
    h.or(h2).or(h3)
}

pub fn scene_panel(ui: &mut egui::Ui, p: &mut SceneParams) {
    // One bar, like 2dini's Merge: it grows a free input as wires land.
    let wired = p.input_count().saturating_sub(1);
    ui.weak(format!("blends {wired} of up to {MAX_SCENE_INPUTS} cards or scenes, nearest in front"));
}

/// `status`: what the scene on its ports came to, for the panel's last
/// line (cards drawn, cards left out, whether a camera is wired).
pub fn render_panel(ui: &mut egui::Ui, rcx: &mut RowCtx, node: u64, p: &mut Render3dParams, status: &str) -> Option<HudSlider> {
    widgets::gen_format_row(ui, ("render3d-format", node), &mut p.size, &mut p.inherit);
    let mut n = p.samples as f32;
    widgets::plain_slider(ui, "blur samples", &mut n, 1.0, MAX_SAMPLES as f32, true);
    p.samples = (n.round() as u32).clamp(1, MAX_SAMPLES as u32);
    let h = widgets::slider(ui, rcx, "shutter", &mut p.shutter, 0.0, 2.0, &[Channel::Render3dShutter]);
    let h2 = widgets::slider(ui, rcx, "offset", &mut p.offset, -0.5, 0.5, &[Channel::Render3dOffset]);
    ui.weak(status);
    h.or(h2)
}

/// One line on what a Render3D's ports come to: cards drawn, cards past
/// the cap, a missing camera. Read off the editor graph, through dots.
pub fn render_status(eg: &crate::editor::EditorGraph, render: u64) -> String {
    use engine::graph::NodeKind;
    use engine::nodes::render3d::{CAMERA_PORT, SCENE_PORT};
    let at = |n: u64, i: usize| eg.input_of(n, i).and_then(|s| eg.resolve_dot(s));
    let kind = |n: u64| eg.node(n).map(|e| &e.kind);
    let cam = at(render, CAMERA_PORT).is_some_and(|c| matches!(kind(c), Some(NodeKind::Camera3D(_))));
    let mut seen = Vec::new();
    let mut cards = 0usize;
    let mut blank = 0usize;
    let mut stack: Vec<u64> = at(render, SCENE_PORT).into_iter().collect();
    while let Some(n) = stack.pop() {
        if seen.contains(&n) {
            continue;
        }
        seen.push(n);
        match kind(n) {
            Some(NodeKind::Card3D(_)) => {
                if at(n, 0).is_some() { cards += 1 } else { blank += 1 }
            }
            Some(NodeKind::Scene3D(p)) => {
                for i in (0..p.input_count()).rev() {
                    stack.extend(at(n, i));
                }
            }
            _ => {}
        }
    }
    let mut out = if !cam {
        "wire a Camera into the first port".to_string()
    } else if cards + blank == 0 {
        "wire a Card or a Blend 3D into the second port".to_string()
    } else {
        format!("{} card{}", cards.min(engine::scene3d::MAX_CARDS), if cards == 1 { "" } else { "s" })
    };
    if cards > engine::scene3d::MAX_CARDS {
        out += &format!(", {} left out ({} at most)", cards - engine::scene3d::MAX_CARDS, engine::scene3d::MAX_CARDS);
    }
    if blank > 0 {
        out += &format!(", {blank} with no picture");
    }
    out
}

/// A camera body with its lens.
pub fn camera_glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let body = egui::Rect::from_center_size(c - Vec2::new(r * 0.2, 0.0), Vec2::new(r * 1.0, r * 0.8));
    painter.rect(body, r * 0.12, fill, stroke, egui::StrokeKind::Middle);
    let lens = vec![
        Pos2::new(body.right(), c.y - r * 0.18),
        Pos2::new(body.right() + r * 0.5, c.y - r * 0.4),
        Pos2::new(body.right() + r * 0.5, c.y + r * 0.4),
        Pos2::new(body.right(), c.y + r * 0.18),
    ];
    painter.add(egui::Shape::convex_polygon(lens, fill, stroke));
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    painter.circle_stroke(body.center(), r * 0.22, ink);
}

/// A card turned in space: a slanted quad.
pub fn card_glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let q = vec![
        c + Vec2::new(-r * 0.55, -r * 0.45),
        c + Vec2::new(r * 0.45, -r * 0.7),
        c + Vec2::new(r * 0.55, r * 0.45),
        c + Vec2::new(-r * 0.45, r * 0.7),
    ];
    painter.add(egui::Shape::convex_polygon(q, fill, stroke));
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    painter.line_segment([c + Vec2::new(-r * 0.3, r * 0.25), c + Vec2::new(r * 0.05, -r * 0.15)], ink);
    painter.line_segment([c + Vec2::new(r * 0.05, -r * 0.15), c + Vec2::new(r * 0.3, r * 0.1)], ink);
}

/// Three cards, one behind another.
pub fn scene_glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    for k in (0..3).rev() {
        let o = Vec2::new(k as f32 * r * 0.22, -(k as f32) * r * 0.22);
        let rect = egui::Rect::from_center_size(c + o - Vec2::new(r * 0.22, -r * 0.22), Vec2::new(r * 0.8, r * 0.8));
        painter.rect(rect, r * 0.08, fill.gamma_multiply(1.0 - 0.25 * k as f32), stroke, egui::StrokeKind::Middle);
    }
}

/// A frustum opening onto a frame.
pub fn render_glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let eye = c - Vec2::new(r * 0.6, 0.0);
    let frame = egui::Rect::from_center_size(c + Vec2::new(r * 0.3, 0.0), Vec2::new(r * 0.6, r * 1.1));
    painter.add(egui::Shape::convex_polygon(
        vec![eye, frame.left_top(), frame.left_bottom()],
        fill.gamma_multiply(0.5),
        stroke,
    ));
    painter.rect(frame, r * 0.06, fill, stroke, egui::StrokeKind::Middle);
    painter.circle_filled(eye, r * 0.12, crate::theme::node_ink());
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    fn fits(name: &str, tiles: usize, ekind: NodeKind, f: impl Fn(&mut egui::Ui, &mut RowCtx)) {
        let mut anim = Default::default();
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let t = crate::widgets::TILE;
        let w = t * tiles as f32 + crate::widgets::TILE_GAP * (tiles as f32 - 1.0);
        let shot = crate::snapshot::shoot(w - 14.0, t - 14.0 - 22.0, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                f(ui, &mut rcx);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save(name);
        assert!(wide.get() <= 0.5, "{name} overflows SIDEWAYS by {}pt", wide.get());
        // Tall panels scroll inside their tile; what must not happen is a
        // row too wide to reach.
        let _ = over.get();
    }

    #[test]
    fn the_panels_fit_their_width() {
        fits("camera3d-panel", CAMERA_TILES, NodeKind::Camera3D(Default::default()), |ui, rcx| {
            let mut p = CameraParams::default();
            let _ = camera_panel(ui, rcx, &mut p, (0, 10), &[], &[]);
        });
        fits("card3d-panel", CARD_TILES, NodeKind::Card3D(Default::default()), |ui, rcx| {
            let mut p = CardParams::default();
            let _ = card_panel(ui, rcx, &mut p, (0, 10), &[]);
        });
        fits("render3d-panel", RENDER_TILES, NodeKind::Render3D(Default::default()), |ui, rcx| {
            let mut p = Render3dParams::default();
            let _ = render_panel(ui, rcx, 1, &mut p, "2 cards");
        });
    }

    #[test]
    fn every_channel_round_trips() {
        let mut cam = CameraParams::default();
        for (k, ch) in CAMERA_CHANNELS.iter().enumerate() {
            camera_set(&mut cam, *ch, k as f32 + 0.5);
            assert_eq!(camera_get(&cam, *ch), Some(k as f32 + 0.5), "{ch:?}");
        }
        let mut card = CardParams::default();
        for (k, ch) in CARD_CHANNELS.iter().enumerate() {
            card_set(&mut card, *ch, k as f32 + 0.5);
            assert_eq!(card_get(&card, *ch), Some(k as f32 + 0.5), "{ch:?}");
        }
        for ch in CHANNELS {
            assert!(label(*ch).is_some(), "{ch:?} has no label");
        }
    }
}
