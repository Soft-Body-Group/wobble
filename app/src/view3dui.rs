//! The 3D viewport: a Card or a Blend 3D on the viewer is drawn in 3D,
//! through an eye the viewer's hands move — Blender's (manifesto 11): MMB
//! orbit, shift+MMB pan, ctrl+MMB / wheel / pinch dolly, two fingers orbit,
//! the numpad's views, R / Home to frame. The eye and the hands are
//! `imfram::view3d`; this file is the glue.
//!
//! The picture is a real Render3D: `lower` (in `to_engine_graph_at_f`,
//! before `scene3d::expand`) puts a transient camera at the eye and a
//! Render3D over the viewed node, and views it. Nothing is written into
//! the comp — the eye is viewer state, like the 2D pan, so moving it is
//! never an undo step and never touches a render that is not the viewer's.
//! The render is the script's format, so the frame on screen is the one a
//! Render3D in the comp would draw.

use crate::Gfx;
use engine::graph::{Graph, NodeKind, Render3dParams};
use engine::scene3d::{CameraParams, CardParams, Source, Xform};
use imfram::view3d::{Orbit, Up, View};
use std::sync::Mutex;

/// What the viewer's eye is, for the node it is looking at.
#[derive(Clone, Debug, PartialEq)]
pub struct Eye {
    pub node: u64,
    pub cam: CameraParams,
    /// The render's format (the viewed node's stamped format too).
    pub size: (u32, u32),
}

/// The eye the next engine graph is built with. A global for the same
/// reason as `scene3d::set_blender_scene`: the graph is built in many
/// places, all of which should see the one viewer eye.
static EYE: Mutex<Option<Eye>> = Mutex::new(None);

pub fn publish(e: Option<Eye>) {
    if let Ok(mut g) = EYE.lock() {
        *g = e;
    }
}

pub fn current() -> Option<Eye> {
    EYE.lock().ok().and_then(|g| g.clone())
}

/// A node the viewer shows in 3D.
pub fn shows_3d(k: &NodeKind) -> bool {
    matches!(k, NodeKind::Card3D(_) | NodeKind::Scene3D(_))
}

/// The camera standing at the eye, its 36mm back across the picture's
/// larger side — exactly the projection `Orbit::project` draws the grid
/// with, so the overlay sits on the render.
pub fn camera(o: &Orbit, size: (u32, u32)) -> CameraParams {
    let long = size.0.max(size.1).max(1) as f32;
    CameraParams {
        xform: Xform {
            translate: o.eye(),
            rotate: engine::scene3d::euler_of(&imfram::view3d::qmat(o.rot)),
            scale: [1.0; 3],
        },
        focal: o.lens,
        aperture: [36.0 * size.0 as f32 / long, 36.0 * size.1 as f32 / long],
        near: (o.dist * 1e-4).max(1e-4),
        far: (o.dist * 1e4).max(1e4),
        source: Source::Wobble,
    }
}

/// Swap a viewed Card / Blend 3D for a Render3D through the eye. `floor`
/// is the highest id the editor uses, so the minted ids collide with
/// nothing the motion-blur pose lookup might find.
pub fn lower(g: &mut Graph, floor: u64) {
    let Some(v) = g.view else { return };
    if !g.nodes.get(&v).is_some_and(shows_3d) {
        return;
    }
    let Some(e) = current().filter(|e| e.node == v) else { return };
    let cam = g.nodes.keys().copied().max().unwrap_or(0).max(floor) + 1;
    let r = cam + 1;
    g.nodes.insert(cam, NodeKind::Camera3D(e.cam));
    g.nodes.insert(
        r,
        NodeKind::Render3D(Render3dParams { size: e.size, inherit: false, samples: 1, ..Default::default() }),
    );
    g.connect_out(r, engine::nodes::render3d::CAMERA_PORT, cam, 0);
    g.connect_out(r, engine::nodes::render3d::SCENE_PORT, v, 0);
    g.format_alias.insert(r, v);
    g.view = Some(r);
    g.view_out = 0;
}

impl Gfx {
    /// The node on the viewer that is shown in 3D, if it is one.
    pub(crate) fn view3d_target(&self) -> Option<u64> {
        let id = self.egraph.active_view()?;
        self.egraph.node(id).is_some_and(|n| shows_3d(&n.kind)).then_some(id)
    }

    /// The 3D view's render size: the script's format at the proxy scale.
    fn view3d_size(&self) -> (u32, u32) {
        self.desired_output().unwrap_or((1920, 1080))
    }

    /// The eye for `id`, made (and framed) on first sight.
    pub(crate) fn view3d_eye(&mut self, id: u64) -> Orbit {
        if let Some(o) = self.view3d.get(&id) {
            return *o;
        }
        let mut o = Orbit::new(Up::Y);
        o.snap(View::Front);
        // A little above and to the side, like Blender's startup view.
        o.turn(0.35, -0.3);
        if let Some((c, r)) = self.view3d_bounds(id) {
            let (w, h) = self.view3d_size();
            o.frame(c, r, egui::vec2(w as f32, h as f32));
        }
        self.view3d.insert(id, o);
        o
    }

    /// Hand the engine graph builder the eye for what is on the viewer
    /// (or none). Called at the top of `ui()` and after every move.
    pub(crate) fn view3d_publish(&mut self) {
        let e = self.view3d_target().map(|id| {
            let o = self.view3d_eye(id);
            let size = self.view3d_size();
            Eye { node: id, cam: camera(&o, size), size }
        });
        if current() != e {
            publish(e);
        }
    }

    /// Frame every card under the viewed node (Home / R).
    pub(crate) fn view3d_frame(&mut self, id: u64) {
        let mut o = self.view3d_eye(id);
        if let Some((c, r)) = self.view3d_bounds(id) {
            let (w, h) = self.view3d_size();
            o.frame(c, r, egui::vec2(w as f32, h as f32));
        }
        self.view3d.insert(id, o);
        self.view3d_publish();
    }

    /// Every card the node reaches, as it stands at the playhead, with
    /// its corners in the world (the unit square at the picture's aspect).
    pub(crate) fn view3d_cards(&self, id: u64) -> Vec<(u64, CardParams, [[f32; 3]; 4])> {
        let mut cards = Vec::new();
        let mut stack = vec![id];
        let mut seen = std::collections::HashSet::new();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            match self.egraph.node(n).map(|n| &n.kind) {
                Some(NodeKind::Card3D(_)) => cards.push(n),
                Some(NodeKind::Scene3D(p)) => {
                    stack.extend((0..p.input_count()).filter_map(|i| self.egraph.input_of(n, i)));
                }
                _ => {}
            }
        }
        cards.sort_unstable();
        let t = self.timeline_pos;
        let mut out = Vec::new();
        for c in cards {
            let Some(NodeKind::Card3D(p)) = self.live_kind(c, t) else { continue };
            let aspect = if p.aspect > 0.0 {
                p.aspect
            } else {
                self.egraph
                    .input_of(c, 0)
                    .and_then(|s| self.node_resolution(s, &mut std::collections::HashSet::new()))
                    .map_or(1.0, |(w, h)| w as f32 / h.max(1) as f32)
            };
            let corners = card_corners(&p.xform, aspect);
            out.push((c, p, corners));
        }
        out
    }

    /// Every camera the node reaches (a Camera wired into a Blend 3D), as
    /// it stands at the playhead with its source applied, so a tracked
    /// camera sits where the solve puts it.
    pub(crate) fn view3d_cameras(&self, id: u64) -> Vec<(u64, CameraParams)> {
        let mut cams = Vec::new();
        let mut stack = vec![id];
        let mut seen = std::collections::HashSet::new();
        while let Some(n) = stack.pop() {
            if !seen.insert(n) {
                continue;
            }
            match self.egraph.node(n).map(|n| &n.kind) {
                Some(NodeKind::Camera3D(_)) => cams.push(n),
                Some(NodeKind::Scene3D(p)) => {
                    stack.extend((0..p.input_count()).filter_map(|i| self.egraph.input_of(n, i)));
                }
                _ => {}
            }
        }
        cams.sort_unstable();
        let t = self.timeline_pos;
        let mut out = Vec::new();
        for c in cams {
            let Some(mut k) = self.live_kind(c, t) else { continue };
            let _ = engine::scene3d::apply_file_source(&mut k, t as f32);
            if let NodeKind::Camera3D(p) = k {
                out.push((c, p));
            }
        }
        out
    }

    /// A sphere round every card and camera the node reaches, at the
    /// playhead.
    fn view3d_bounds(&self, id: u64) -> Option<([f32; 3], f32)> {
        let mut pts: Vec<[f32; 3]> = self.view3d_cards(id).into_iter().flat_map(|(_, _, c)| c).collect();
        for (_, p) in self.view3d_cameras(id) {
            pts.push(p.xform.translate);
            // A tracked camera brings its path and its cloud (a sample of
            // it: the bounds want the spread, not every point).
            if let Source::Tracker { node } = &p.source {
                if let Some(t) = engine::scene3d::tracker_solve(*node) {
                    let s = &t.solve.solve;
                    for i in 0..s.poses.len() {
                        let m = t.camera_mat(i);
                        pts.push([m[0][3], m[1][3], m[2][3]]);
                    }
                    let step = (s.points.len() / 400).max(1);
                    pts.extend((0..s.points.len()).step_by(step).map(|k| t.point_world(k)));
                }
            }
        }
        if pts.is_empty() {
            return None;
        }
        let n = pts.len() as f32;
        let c = pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n]);
        let r = pts
            .iter()
            .map(|p| ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt())
            .fold(0.0f32, f32::max);
        Some((c, r.max(0.05)))
    }

    /// The viewer's hands on the eye, this frame; true when it moved.
    pub(crate) fn view3d_nav(&mut self, id: u64, ui: &egui::Ui, nav: &egui::Response, rect: egui::Rect) -> bool {
        let mut o = self.view3d_eye(id);
        let moved = o.navigate(ui, nav, rect);
        let keys = ui.input(|i| {
            use crate::keymap::{pressed, Act};
            [
                (Act::View3dFront, 0),
                (Act::View3dBack, 1),
                (Act::View3dRight, 2),
                (Act::View3dLeft, 3),
                (Act::View3dTop, 4),
                (Act::View3dBottom, 5),
                (Act::View3dOrbitLeft, 6),
                (Act::View3dOrbitRight, 7),
                (Act::View3dOrbitUp, 8),
                (Act::View3dOrbitDown, 9),
                (Act::View3dOpposite, 10),
            ]
            .into_iter()
            .filter(|(a, _)| pressed(i, *a))
            .map(|(_, k)| k)
            .collect::<Vec<_>>()
        });
        let keyed = nav.hovered() && !keys.is_empty() && !ui.ctx().egui_wants_keyboard_input();
        if keyed {
            use imfram::view3d::Step;
            for k in keys {
                match k {
                    0 => o.snap(View::Front),
                    1 => o.snap(View::Back),
                    2 => o.snap(View::Right),
                    3 => o.snap(View::Left),
                    4 => o.snap(View::Top),
                    5 => o.snap(View::Bottom),
                    6 => o.step(Step::Left),
                    7 => o.step(Step::Right),
                    8 => o.step(Step::Up),
                    9 => o.step(Step::Down),
                    _ => o.opposite(),
                }
            }
        }
        if moved || keyed {
            self.view3d.insert(id, o);
            self.view3d_publish();
            ui.ctx().request_repaint();
        }
        moved || keyed
    }

    /// What the 3D view draws over its picture: the floor grid, every
    /// card's outline (so a card with no picture is still there to see and
    /// pick), the picked card's manipulator, the move / rotate / scale
    /// buttons and the axis gizmo. A click on a card picks it. Returns the
    /// gestures for the viewer's hotkey strip.
    pub(crate) fn view3d_overlay(
        &mut self,
        id: u64,
        ui: &mut egui::Ui,
        nav: &egui::Response,
        rect: egui::Rect,
        avail: egui::Rect,
    ) -> Vec<crate::hudkeys::HudKey> {
        use imfram::view3d::{self as v3, Tool};
        let style = v3::Style {
            grid: crate::theme::view3d_grid(),
            x: crate::theme::view3d_x(),
            y: crate::theme::view3d_y(),
            z: crate::theme::view3d_z(),
        };
        let mut o = self.view3d_eye(id);
        let painter = ui.painter_at(rect);
        v3::paint_grid(&painter, rect, &o, &style);
        let cards = self.view3d_cards(id);
        // The card the manipulator sits on: the picked one, else the
        // viewed card itself.
        let picked = self
            .selected
            .iter()
            .copied()
            .find(|s| cards.iter().any(|c| c.0 == *s))
            .or_else(|| cards.iter().find(|c| c.0 == id).map(|c| c.0));
        for (c, _, corners) in &cards {
            let col = if Some(*c) == picked { crate::theme::view3d_card_picked() } else { crate::theme::view3d_card() };
            v3::paint_loop(&painter, rect, &o, corners, egui::Stroke::new(1.5, col));
        }
        // Every camera in the scene: a frustum where it stands; a tracked
        // one also its whole path and the points it solved.
        for (cam, p) in self.view3d_cameras(id) {
            paint_camera(&painter, rect, &o, &p, o.dist * 0.12, Some(cam) == self.selected.first().copied());
            if let Source::Tracker { node } = &p.source {
                if let Some(t) = engine::scene3d::tracker_solve(*node) {
                    paint_track(&painter, rect, &o, &t);
                }
            }
        }
        // Tool keys: Blender's G / R / S, over the view.
        let over = nav.hovered() && !ui.ctx().egui_wants_keyboard_input();
        if over {
            ui.input(|i| {
                let plain = |k| i.key_pressed(k) && i.modifiers.is_none();
                if plain(egui::Key::G) {
                    self.view3d_tool = Tool::Move;
                } else if plain(egui::Key::R) {
                    self.view3d_tool = Tool::Rotate;
                } else if plain(egui::Key::S) {
                    self.view3d_tool = Tool::Scale;
                }
            });
        }
        // The manipulator, on the picked card.
        let mut manip = v3::ManipOut::default();
        if let Some((card, p, _)) = picked.and_then(|k| cards.iter().find(|c| c.0 == k)).cloned() {
            let r = engine::scene3d::rotation(p.xform.rotate);
            let local = [0, 1, 2].map(|c| [r[0][c], r[1][c], r[2][c]]);
            manip = v3::manipulator(ui, rect, &o, self.view3d_tool, p.xform.translate, local, &style);
            if manip.started {
                self.view3d_press = Some((card, p.xform.clone()));
            }
            if let (Some(e), Some((pc, press))) = (manip.edit, self.view3d_press.clone()) {
                if pc == card {
                    self.view3d_apply(card, &press, e);
                }
            }
            if manip.ended {
                self.view3d_press = None;
            }
        }
        // A click on a card picks it (the nearest under the pointer).
        if nav.clicked() && !manip.hovered && !manip.active {
            if let Some(at) = nav.interact_pointer_pos() {
                let hit = cards
                    .iter()
                    .filter_map(|(c, _, k)| v3::hit_loop(rect, &o, k, at).map(|d| (d, *c)))
                    .min_by(|a, b| a.0.total_cmp(&b.0));
                if let Some((_, c)) = hit {
                    self.selected = vec![c];
                }
            }
        }
        // The tool buttons, Blender's toolbar at the view's left.
        let mut y = avail.min.y + 40.0;
        for (tool, label) in [(Tool::Move, "move  G"), (Tool::Rotate, "rotate  R"), (Tool::Scale, "scale  S")] {
            let r = egui::Rect::from_min_size(egui::pos2(avail.min.x + 10.0, y), egui::vec2(78.0, 20.0));
            if ui.put(r, egui::Button::selectable(self.view3d_tool == tool, label)).clicked() {
                self.view3d_tool = tool;
            }
            y += 23.0;
        }
        let gr = 34.0;
        let at = egui::pos2(avail.max.x - gr - 14.0, avail.min.y + gr + 40.0);
        if v3::gizmo(ui, at, gr, &mut o, &style) {
            self.view3d.insert(id, o);
            self.view3d_publish();
            ui.ctx().request_repaint();
        }
        use crate::hudkeys::{chord, key};
        vec![
            chord("MMB", "orbit · shift pan · ctrl dolly"),
            key("G", "move", egui::Key::G),
            key("R", "rotate", egui::Key::R),
            key("S", "scale", egui::Key::S),
            chord("Home", "frame"),
            chord("1 3 7", "views"),
        ]
    }

    /// A manipulator's edit, measured from its press, applied to the card
    /// as it was then: absolute, so an animated card keys cleanly (the
    /// gizmo trap in CLAUDE.md), through the panel rows' keyed write-back.
    fn view3d_apply(&mut self, card: u64, press: &Xform, e: imfram::view3d::Edit) {
        use imfram::view3d::Edit;
        let mut x = press.clone();
        match e {
            Edit::Move(d) => {
                for i in 0..3 {
                    x.translate[i] = press.translate[i] + d[i];
                }
            }
            Edit::Rotate { axis, angle } => {
                let turn = imfram::view3d::qmat(imfram::view3d::qaxis(axis, angle));
                let r0 = engine::scene3d::rotation(press.rotate);
                let mut m = [[0.0f32; 3]; 3];
                for (r, row) in m.iter_mut().enumerate() {
                    for (c, v) in row.iter_mut().enumerate() {
                        *v = (0..3).map(|k| turn[r][k] * r0[k][c]).sum();
                    }
                }
                x.rotate = nearest_euler(engine::scene3d::euler_of(&m), press.rotate);
            }
            Edit::Scale(f) => {
                for i in 0..3 {
                    x.scale[i] = press.scale[i] * f[i];
                }
            }
        }
        let t = self.eval_tick();
        let autokey = self.autokey;
        let driven: Vec<(u64, crate::anim::Channel)> = self
            .egraph
            .patch
            .links
            .iter()
            .filter_map(|l| match l.to {
                crate::editor::PDst::Chan(n, ch) => Some((n, ch)),
                _ => None,
            })
            .collect();
        if let Some(n) = self.egraph.node_mut(card) {
            let crate::editor::ENode { kind, anim, .. } = n;
            let before = kind.clone();
            let mut after = kind.clone();
            if let NodeKind::Card3D(p) = &mut after {
                p.xform = x;
            }
            crate::keyed_write_back(
                anim,
                kind,
                &before,
                &after,
                t,
                &crate::nodes::scene3d::CARD_CHANNELS[..9],
                autokey,
                &driven,
                card,
            );
        }
    }
}

/// A camera as a frustum: its origin, four rays to the picture `d` units
/// in front (its focal and filmback give the angle) and that picture's
/// frame, with a mark on the top edge for which way is up.
pub fn paint_camera(painter: &egui::Painter, rect: egui::Rect, o: &Orbit, p: &CameraParams, d: f32, picked: bool) {
    let col = if picked { crate::theme::view3d_card_picked() } else { crate::theme::view3d_camera() };
    let stroke = egui::Stroke::new(1.2, col);
    let m = p.xform.matrix();
    let world = |l: [f32; 3]| -> [f32; 3] {
        std::array::from_fn(|i| m[i][0] * l[0] + m[i][1] * l[1] + m[i][2] * l[2] + m[i][3])
    };
    let f = p.focal.max(1e-3);
    let (hw, hh) = (d * p.aperture[0] * 0.5 / f, d * p.aperture[1].max(1e-3) * 0.5 / f);
    let eye = world([0.0, 0.0, 0.0]);
    let corners = [[-hw, hh, -d], [hw, hh, -d], [hw, -hh, -d], [-hw, -hh, -d]].map(world);
    imfram::view3d::paint_loop(painter, rect, o, &corners, stroke);
    for c in &corners {
        if let (Some(a), Some(b)) = (o.project(eye, rect), o.project(*c, rect)) {
            painter.line_segment([a, b], stroke);
        }
    }
    let up = [world([-hw * 0.4, hh, -d]), world([0.0, hh * 1.6, -d]), world([hw * 0.4, hh, -d])];
    imfram::view3d::paint_loop(painter, rect, o, &up, stroke);
}

/// A tracked camera's path (a line through every solved frame) and its
/// point cloud.
pub fn paint_track(painter: &egui::Painter, rect: egui::Rect, o: &Orbit, t: &engine::graph::CamTrackParams) {
    let s = &t.solve.solve;
    let path: Vec<egui::Pos2> = (0..s.poses.len())
        .filter_map(|i| {
            let m = t.camera_mat(i);
            o.project([m[0][3], m[1][3], m[2][3]], rect)
        })
        .collect();
    if path.len() >= 2 {
        painter.add(egui::Shape::line(path, egui::Stroke::new(2.0, crate::theme::view3d_camera())));
    }
    let col = crate::theme::view3d_points();
    let picks = t.picks();
    let mut mesh = egui::epaint::Mesh::default();
    for k in 0..s.points.len() {
        let Some(c) = o.project(t.point_world(k), rect) else { continue };
        if picks.contains(&k) {
            painter.circle_filled(c, 4.0, crate::theme::view3d_card_picked());
            continue;
        }
        let i = mesh.vertices.len() as u32;
        for q in [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]] {
            mesh.vertices.push(egui::epaint::Vertex {
                pos: egui::pos2(c.x + q[0], c.y + q[1]),
                uv: egui::epaint::WHITE_UV,
                color: col,
            });
        }
        mesh.indices.extend_from_slice(&[i, i + 1, i + 2, i, i + 2, i + 3]);
    }
    if !mesh.is_empty() {
        painter.add(egui::Shape::mesh(mesh));
    }
}

/// A card's corners in the world: the unit-wide square at `aspect`
/// (width over height), through its transform.
pub fn card_corners(x: &Xform, aspect: f32) -> [[f32; 3]; 4] {
    let m = x.matrix();
    let hy = 0.5 / aspect.max(1e-3);
    [(-0.5, -hy), (0.5, -hy), (0.5, hy), (-0.5, hy)].map(|(a, b)| {
        [
            m[0][0] * a + m[0][1] * b + m[0][3],
            m[1][0] * a + m[1][1] * b + m[1][3],
            m[2][0] * a + m[2][1] * b + m[2][3],
        ]
    })
}

/// The Euler triple for the same turn nearest `near`, so a rotate drag
/// never jumps a channel by 360° (or flips through the other solution).
pub fn nearest_euler(e: [f32; 3], near: [f32; 3]) -> [f32; 3] {
    let wrap = |v: f32, n: f32| v + ((n - v) / 360.0).round() * 360.0;
    let a = [0, 1, 2].map(|i| wrap(e[i], near[i]));
    let alt = [e[0] + 180.0, 180.0 - e[1], e[2] + 180.0];
    let b = [0, 1, 2].map(|i| wrap(alt[i], near[i]));
    let d = |x: [f32; 3]| (0..3).map(|i| (x[i] - near[i]).abs()).sum::<f32>();
    if d(b) < d(a) {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The eye's camera, rendered, puts a world point where the grid
    /// overlay draws it: the engine's projection against imfram's.
    #[test]
    fn the_camera_projects_where_the_overlay_draws() {
        let mut o = Orbit::new(Up::Y);
        o.turn(0.4, -0.3);
        o.pivot = [0.3, -0.2, -1.0];
        o.dist = 3.0;
        let size = (1920u32, 1080u32);
        let cam = camera(&o, size);
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size.0 as f32, size.1 as f32));
        for p in [[0.0, 0.0, 0.0], [0.5, 0.25, -1.5], [-0.4, -0.6, -0.2]] {
            let want = o.project(p, rect).unwrap();
            // The engine's camera: world -> camera space by the inverse of
            // its matrix, looking down -Z, focal_px across the width.
            let inv = cam.xform.inverse_scaled([1.0; 3]);
            let v = [0, 1, 2].map(|r| inv[r][0] * p[0] + inv[r][1] * p[1] + inv[r][2] * p[2] + inv[r][3]);
            let f = cam.focal_px(size.0 as f32);
            let got = egui::pos2(size.0 as f32 / 2.0 + f * v[0] / -v[2], size.1 as f32 / 2.0 - f * v[1] / -v[2]);
            assert!((got - want).length() < 0.05, "{p:?}: engine {got:?}, overlay {want:?}");
        }
    }

    #[test]
    fn the_nearest_euler_is_the_same_turn_and_never_jumps() {
        let same = |a: [f32; 3], b: [f32; 3]| {
            let (ra, rb) = (engine::scene3d::rotation(a), engine::scene3d::rotation(b));
            (0..3).all(|r| (0..3).all(|c| (ra[r][c] - rb[r][c]).abs() < 1e-4))
        };
        let near = [10.0, 170.0, 350.0];
        let e = engine::scene3d::euler_of(&engine::scene3d::rotation([12.0, 168.0, -8.0]));
        let n = nearest_euler(e, near);
        assert!(same(n, e), "{n:?} is not the turn {e:?}");
        assert!((0..3).all(|i| (n[i] - near[i]).abs() < 20.0), "{n:?} jumped from {near:?}");
    }

    #[test]
    fn a_viewed_card_lowers_to_a_render_through_the_eye() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::Card3D(Default::default()));
        g.view = Some(1);
        let o = Orbit::new(Up::Y);
        publish(Some(Eye { node: 1, cam: camera(&o, (64, 32)), size: (64, 32) }));
        lower(&mut g, 5);
        publish(None);
        let r = g.view.unwrap();
        assert!(matches!(g.nodes.get(&r), Some(NodeKind::Render3D(p)) if p.size == (64, 32)));
        assert_eq!(g.input_of(r, engine::nodes::render3d::SCENE_PORT), Some(1));
        let cam = g.input_of(r, engine::nodes::render3d::CAMERA_PORT).unwrap();
        assert!(cam > 5 && matches!(g.nodes.get(&cam), Some(NodeKind::Camera3D(_))));
        assert_eq!(g.format_alias.get(&r), Some(&1));
    }
}
