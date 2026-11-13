//! The PlanarTrack node's app side: the bake job, the properties panel and
//! the viewer gizmo. The maths is `engine::planar`; this file feeds it
//! frames and draws what it says.
//!
//! Like the blob Tracker this is an explicit BAKE on a worker thread: the
//! job decodes the input sequence from the frame the region was drawn on,
//! forward to the end or backward to the start, and streams one
//! `PlanarFrame` per frame back. The app writes them onto the node's
//! `track` data as they arrive, so the overlay fills in while it runs.

use engine::decode::{ExrDecoder, ImageInfo};
use engine::diskcache::DiskCache;
use engine::graph::{PlanarFrame, PlanarTrackParams};
use engine::planar::{self, Mesh, MeshOpts, Plane, TrackOpts, Tracker};
use half::f16;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::anim::Channel;
use crate::ink;
use crate::widgets;

/// What the panel asked for, applied once the node borrow ends.
pub enum PlanarAction {
    Start { backward: bool },
    Cancel,
    /// Cable the track's corner pin into a node that can follow it.
    Drive(DriveTarget),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriveTarget {
    Transform,
    CornerPin,
    /// A CornerPin whose FROM quad follows the track and whose TO stays the
    /// frame: from -> to is the inverse of the tracked motion, so the
    /// tracked region holds still (a lock-off). At the reference frame the
    /// track's pin IS the frame, so that frame passes through unchanged.
    Stabilize,
}

/// The eight channels of the target a source's pin (corner 0 x, y … corner
/// 3 x, y) is cabled into.
pub fn drive_channels(target: DriveTarget) -> [Channel; 8] {
    match target {
        DriveTarget::Transform | DriveTarget::CornerPin => [
            Channel::Pin0X,
            Channel::Pin0Y,
            Channel::Pin1X,
            Channel::Pin1Y,
            Channel::Pin2X,
            Channel::Pin2Y,
            Channel::Pin3X,
            Channel::Pin3Y,
        ],
        DriveTarget::Stabilize => [
            Channel::From0X,
            Channel::From0Y,
            Channel::From1X,
            Channel::From1Y,
            Channel::From2X,
            Channel::From2Y,
            Channel::From3X,
            Channel::From3Y,
        ],
    }
}

/// The channels the bake keys, in the order `drive_of` returns them.
pub const DRIVE_CHANNELS: [Channel; 4] =
    [Channel::PlanarDX, Channel::PlanarDY, Channel::PlanarRot, Channel::PlanarScale];
pub const PIN_CHANNELS: [Channel; 8] = [
    Channel::PlanarPin0X,
    Channel::PlanarPin0Y,
    Channel::PlanarPin1X,
    Channel::PlanarPin1Y,
    Channel::PlanarPin2X,
    Channel::PlanarPin2Y,
    Channel::PlanarPin3X,
    Channel::PlanarPin3Y,
];

/// What a frame's result says to a Transform. The similarity (dx, dy in
/// pixels, rotation in degrees, scale) that best carries the region's
/// reference corners onto the tracked ones — a least-squares fit, in the
/// Transform's own convention: rotate and scale about the reference
/// centroid, then move by the centroids' difference. And the frame's own
/// corners under the frame's homography, normalised: a corner pin.
pub fn drive_of(
    reference: &[[f32; 2]; 4],
    corners: &[[f32; 2]; 4],
    size: (f32, f32),
) -> ([f32; 4], [[f32; 2]; 4]) {
    let (w, h) = (size.0 as f64, size.1 as f64);
    let px = |c: &[[f32; 2]; 4]| -> [[f64; 2]; 4] {
        let mut o = [[0.0; 2]; 4];
        for k in 0..4 {
            o[k] = [c[k][0] as f64 * w, c[k][1] as f64 * h];
        }
        o
    };
    let src = px(reference);
    let dst = px(corners);
    let cen = |q: &[[f64; 2]; 4]| [q.iter().map(|p| p[0]).sum::<f64>() / 4.0, q.iter().map(|p| p[1]).sum::<f64>() / 4.0];
    let cs = cen(&src);
    let cd = cen(&dst);
    // Points as complex numbers: q ≈ z·p, z = Σ conj(p)·q / Σ|p|².
    let (mut a, mut b, mut pp) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..4 {
        let p = [src[k][0] - cs[0], src[k][1] - cs[1]];
        let q = [dst[k][0] - cd[0], dst[k][1] - cd[1]];
        a += p[0] * q[0] + p[1] * q[1];
        b += p[0] * q[1] - p[1] * q[0];
        pp += p[0] * p[0] + p[1] * p[1];
    }
    let drive = if pp > 1e-9 {
        [
            (cd[0] - cs[0]) as f32,
            (cd[1] - cs[1]) as f32,
            b.atan2(a).to_degrees() as f32,
            ((a * a + b * b).sqrt() / pp) as f32,
        ]
    } else {
        engine::graph::DRIVE_REST
    };
    let pin = match planar::h_from_quad(&src, &dst) {
        Some(hh) => {
            let mut pin = [[0.0f32; 2]; 4];
            for (k, c) in engine::graph::UNIT_QUAD.iter().enumerate() {
                let (x, y) = planar::h_apply(&hh, c[0] as f64 * w, c[1] as f64 * h);
                pin[k] = [(x / w) as f32, (y / h) as f32];
            }
            pin
        }
        None => engine::graph::UNIT_QUAD,
    };
    (drive, pin)
}

enum Msg {
    /// The mesh topology (vertices normalised at the reference frame).
    Mesh(Mesh),
    Frame(PlanarFrame),
    Done,
    Error(String),
}

/// A running track, as the panel reports it.
#[derive(Clone, Copy, Debug)]
pub struct JobState {
    pub done: u64,
    pub total: u64,
    pub lost: u64,
    pub reanchors: u64,
    /// Comp frame the track was seeded from.
    pub start: i64,
    pub backward: bool,
}

pub struct PlanarJob {
    rx: std::sync::mpsc::Receiver<Msg>,
    cancel: Arc<AtomicBool>,
    state: JobState,
    pub error: Option<String>,
    pub finished: bool,
}

impl Drop for PlanarJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// What a poll hands the app.
#[derive(Default)]
pub struct Polled {
    pub mesh: Option<Mesh>,
    pub frames: Vec<PlanarFrame>,
}

impl PlanarJob {
    pub fn poll(&mut self) -> Polled {
        let mut out = Polled::default();
        for msg in self.rx.try_iter() {
            match msg {
                Msg::Mesh(m) => out.mesh = Some(m),
                Msg::Frame(f) => {
                    self.state.done += 1;
                    if f.lost {
                        self.state.lost += 1;
                    }
                    if f.reanchored {
                        self.state.reanchors += 1;
                    }
                    out.frames.push(f);
                }
                Msg::Done => self.finished = true,
                Msg::Error(e) => {
                    self.error = Some(e);
                    self.finished = true;
                }
            }
        }
        out
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn state(&self) -> JobState {
        self.state
    }
}

/// Kick a track over `frames` (the input's windowed sequence; index 0
/// plays at comp frame `tick0`), seeded from index `ref_idx`, running to
/// the end of the list or, `backward`, to its start.
#[allow(clippy::too_many_arguments)]
pub fn spawn(
    decoder: Arc<dyn ExrDecoder>,
    cache: Option<Arc<DiskCache>>,
    frames: Vec<PathBuf>,
    info: ImageInfo,
    ref_idx: usize,
    tick0: i64,
    backward: bool,
    params: &PlanarTrackParams,
) -> PlanarJob {
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let n = frames.len();
    let ref_idx = ref_idx.min(n.saturating_sub(1));
    let total = if backward { ref_idx as u64 + 1 } else { (n - ref_idx) as u64 };
    let stop = cancel.clone();
    let (w, h) = (info.width as usize, info.height as usize);
    let polygon: Vec<[f64; 2]> =
        params.corners.iter().map(|c| [c[0] as f64 * w as f64, c[1] as f64 * h as f64]).collect();
    let opts = TrackOpts {
        model: params.model,
        reanchor_below: params.reanchor,
        mesh: params.mesh.then_some(MeshOpts {
            cells: params.mesh_cells.max(1) as usize,
            patch: params.mesh_patch.max(3) as usize,
            smooth: params.mesh_smooth,
        }),
        detail: params.detail.then_some(planar::DETAIL_RADIUS),
        ..Default::default()
    };
    crate::bg::spawn(move || {
        let stride = info.row_stride_f16();
        let mut buf = vec![f16::ZERO; stride * h];
        let load = |i: usize, buf: &mut Vec<f16>| -> anyhow::Result<Plane> {
            // The NVMe tier first: a frame the viewer has shown is a raw
            // read, not a decode (a 168 MB multi-part EXR takes ~0.6 s to
            // decode and ~30 ms to read back). Same key the player writes.
            let cached = cache.as_ref().and_then(|c| {
                let file = c.frame_file_layer_scaled(&frames[i], None, 0, 1)?;
                DiskCache::read_frame_into(&file, info, buf).ok().filter(|&ok| ok)
            });
            if cached.is_none() {
                decoder.decode_into(&frames[i], info, stride, buf)?;
            }
            Ok(Plane::from_rgba_f16(buf, w, h, stride))
        };
        let reference = match load(ref_idx, &mut buf) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.send(Msg::Error(format!("{}: {e:#}", frames[ref_idx].display())));
                return;
            }
        };
        let mut tracker = Tracker::new(reference, polygon, opts);
        if let Some(m) = tracker.mesh() {
            let mesh = Mesh {
                verts: m.verts.iter().map(|v| [v[0] / w as f32, v[1] / h as f32]).collect(),
                tris: m.tris.clone(),
            };
            let _ = tx.send(Msg::Mesh(mesh));
        }
        let order: Vec<usize> = if backward { (0..=ref_idx).rev().collect() } else { (ref_idx..n).collect() };
        for i in order {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            let frame = match load(i, &mut buf) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{}: {e:#}", frames[i].display())));
                    return;
                }
            };
            let r = tracker.step(frame);
            let corners_px = tracker.corners(&r.h);
            let mut corners = [[0.0f32; 2]; 4];
            for (k, c) in corners_px.iter().take(4).enumerate() {
                corners[k] = [(c[0] / w as f64) as f32, (c[1] / h as f64) as f32];
            }
            let pf = PlanarFrame {
                frame: tick0 + i as i64,
                corners,
                score: r.score,
                lost: r.lost,
                reanchored: r.reanchored,
                mesh: r.mesh.iter().map(|v| [v[0] / w as f32, v[1] / h as f32]).collect(),
            };
            if tx.send(Msg::Frame(pf)).is_err() {
                return; // the node went away
            }
        }
        let _ = tx.send(Msg::Done);
    });
    PlanarJob {
        rx,
        cancel,
        state: JobState { done: 0, total, lost: 0, reanchors: 0, start: tick0 + ref_idx as i64, backward },
        error: None,
        finished: false,
    }
}

/// The properties panel. Edits `p` in place; a track or cancel comes back
/// as an action for the app to run once the node borrow is over.
pub fn panel(
    ui: &mut egui::Ui,
    p: &mut PlanarTrackParams,
    job: Option<&JobState>,
    cur_t: i64,
) -> Option<PlanarAction> {
    let mut action = None;
    let running = job.is_some();
    ui.horizontal(|ui| {
        ui.label("model");
        egui::ComboBox::from_id_salt("planar-model")
            .selected_text(p.model.label())
            .show_ui(ui, |ui| {
                for m in planar::Model::ALL {
                    ui.selectable_value(&mut p.model, m, m.label());
                }
            })
            .response
            .on_hover_text("what the plane may do between frames; perspective is the corner pin");
    });
    widgets::plain_slider(ui, "re-anchor", &mut p.reanchor, 0.0, 1.0, false)
        .on_hover_text(
            "below this match score the template is re-cut from the current frame: \
             0 never (no drift, fails on appearance change), 1 every frame (frame to \
             frame, drifts)",
        );
    ui.checkbox(&mut p.detail, "detail layer").on_hover_text(
        "track the high-passed picture instead of the raw one: try it when a surface's shading \
         changes with the view faster than its texture can hold; costs some accuracy on clean footage",
    );
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.mesh, "mesh")
            .on_hover_text("track a mesh over the region too: the deformation a plane cannot express");
        if p.mesh {
            ui.label("cells");
            crate::widgets::num_field(ui, egui::DragValue::new(&mut p.mesh_cells).update_while_editing(false).speed(widgets::rated(ui, 0.1)).range(2..=60),
            );
            ui.label("patch");
            crate::widgets::num_field(ui, egui::DragValue::new(&mut p.mesh_patch).update_while_editing(false).speed(widgets::rated(ui, 0.1)).range(4..=32),
            )
            .on_hover_text("half-size of each vertex's alignment patch, pixels");
        }
    });
    if p.mesh {
        widgets::plain_slider(ui, "smooth", &mut p.mesh_smooth, 0.0, 2.0, false)
            .on_hover_text("how hard a vertex leans on its neighbours");
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                action = Some(PlanarAction::Cancel);
            }
        } else {
            if ui
                .button("⏴ track")
                .on_hover_text(format!("track from frame {cur_t} back to the start of the clip"))
                .clicked()
            {
                action = Some(PlanarAction::Start { backward: true });
            }
            if ui
                .button("track ⏵")
                .on_hover_text(format!(
                    "an explicit bake: decodes the input from frame {cur_t} to the end and stores a \
                     corner pin (and mesh) per frame on the node"
                ))
                .clicked()
            {
                action = Some(PlanarAction::Start { backward: false });
            }
        }
        if ui.button("clear").on_hover_text("drop the track; the region becomes editable again").clicked() {
            p.track = Default::default();
        }
    });
    ui.horizontal(|ui| {
        ui.label("drive");
        if ui
            .button("a Transform")
            .on_hover_text(
                "cable the track's corner pin into the selected Transform (or a new one): wire your \
                 element into it and it rides the plane, perspective and all; the Transform's own \
                 translate, rotate and scale still place the element on the plane",
            )
            .clicked()
        {
            action = Some(PlanarAction::Drive(DriveTarget::Transform));
        }
        if ui
            .button("a CornerPin")
            .on_hover_text("cable the track's corner pin into the selected CornerPin (or a new one)")
            .clicked()
        {
            action = Some(PlanarAction::Drive(DriveTarget::CornerPin));
        }
        if ui
            .button("stabilise")
            .on_hover_text(
                "cable the track into the FROM corners of the selected CornerPin (or a new one): \
                 wire the plate into it and the tracked region holds still where it sits on the \
                 reference frame",
            )
            .clicked()
        {
            action = Some(PlanarAction::Drive(DriveTarget::Stabilize));
        }
    });
    ui.weak(
        "channels: dx dy rot scale (a similarity since the reference frame: pixels, degrees) and \
         pin 0-3 (the frame's corners under the track) — patch them into anything",
    );
    if let Some(j) = job {
        let frac = if j.total > 0 { j.done as f32 / j.total as f32 } else { 0.0 };
        let dir = if j.backward { "backward" } else { "forward" };
        ui.add(egui::ProgressBar::new(frac).text(format!("tracking {}/{} {dir} from frame {}", j.done, j.total, j.start)));
    }
    let t = &p.track;
    if let Some((a, b)) = t.span() {
        let lost = t.lost();
        let mut line = format!("tracked {a}..{b}");
        if let Some(r) = t.reference {
            line.push_str(&format!(" from {r}"));
        }
        if lost > 0 {
            line.push_str(&format!(", {lost} lost"));
        }
        if !t.mesh.verts.is_empty() {
            line.push_str(&format!(", mesh {} verts", t.mesh.verts.len()));
        }
        ui.weak(line);
        match t.at(cur_t) {
            Some(f) if f.lost => {
                ui.colored_label(crate::theme::warn(), format!("frame {cur_t}: LOST (holding the last plane)"));
            }
            Some(f) => {
                let mut s = format!("frame {cur_t}: score {:.3}", f.score);
                if f.reanchored {
                    s.push_str(" (re-anchored here)");
                }
                ui.weak(s);
            }
            None => {
                ui.weak(format!("frame {cur_t}: not tracked"));
            }
        }
    } else {
        ui.weak("drag the quad onto a flat surface in the viewer, then track");
    }
    ui.weak("CPU dense alignment (ESM), coarse to fine; a slow path by design, and an explicit bake");
    action
}

/// What the viewer draws for the node at frame `t`: the tracked quad and
/// mesh when the frame has a result, else the authored region with handles.
pub struct Overlay {
    pub corners: [[f32; 2]; 4],
    pub result: Option<PlanarFrame>,
    pub tris: Vec<[u32; 3]>,
}

pub fn overlay(p: &PlanarTrackParams, t: i64) -> Overlay {
    Overlay { corners: p.corners, result: p.track.at(t).cloned(), tris: p.track.mesh.tris.clone() }
}

/// The gizmo. `to_scr` maps a normalised position to the screen and
/// `from_screen` a screen point back to the node's PIXELS (`size`). Returns
/// the region's new corners when a handle was dragged.
pub fn gizmo(
    ui: &mut egui::Ui,
    ink: &mut ink::InkShapes,
    node: u64,
    ov: &Overlay,
    size: (f32, f32),
    to_scr: &dyn Fn([f32; 2]) -> egui::Pos2,
    from_screen: &dyn Fn(egui::Pos2) -> egui::Pos2,
    pointer_pos: Option<egui::Pos2>,
) -> Option<[[f32; 2]; 4]> {
    let quad = |ink: &mut ink::InkShapes, cs: &[[f32; 2]; 4], w: f32, alpha: f32| {
        let s: Vec<egui::Pos2> = cs.iter().map(|c| to_scr(*c)).collect();
        for k in 0..4 {
            ink.seg(s[k], s[(k + 1) % 4], w, alpha, false);
        }
    };
    if let Some(r) = &ov.result {
        // The tracked plane: the quad (red when lost) and the mesh.
        if r.lost {
            ink.set_tint(Some([1.0, 0.3, 0.3]));
        }
        quad(ink, &r.corners, 1.4, 0.9);
        let c = to_scr([
            r.corners.iter().map(|c| c[0]).sum::<f32>() / 4.0,
            r.corners.iter().map(|c| c[1]).sum::<f32>() / 4.0,
        ]);
        ink.seg(c - egui::vec2(7.0, 0.0), c + egui::vec2(7.0, 0.0), 1.0, 0.6, false);
        ink.seg(c - egui::vec2(0.0, 7.0), c + egui::vec2(0.0, 7.0), 1.0, 0.6, false);
        if !r.mesh.is_empty() {
            let mut drawn: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
            for t in &ov.tris {
                for k in 0..3 {
                    let (a, b) = (t[k], t[(k + 1) % 3]);
                    let key = (a.min(b), a.max(b));
                    if !drawn.insert(key) {
                        continue;
                    }
                    let (Some(pa), Some(pb)) = (r.mesh.get(a as usize), r.mesh.get(b as usize)) else {
                        continue;
                    };
                    ink.seg(to_scr(*pa), to_scr(*pb), 0.8, 0.45, false);
                }
            }
        }
        return None;
    }
    // No result here: the region, editable. Corners drag freely (it is a
    // quad, not a box), the centre moves the whole thing.
    let mut cs = ov.corners;
    let before = cs;
    quad(ink, &cs, 1.2, 0.85);
    let centre = [
        cs.iter().map(|c| c[0]).sum::<f32>() / 4.0,
        cs.iter().map(|c| c[1]).sum::<f32>() / 4.0,
    ];
    let cp = to_scr(centre);
    let mv = ui.interact(
        egui::Rect::from_center_size(cp, egui::vec2(24.0, 24.0)),
        ui.id().with(("planar-move", node)),
        egui::Sense::drag(),
    );
    ink.circle_fill(cp, 3.5, 0.9, mv.hovered());
    if mv.dragged() && !mv.drag_started() {
        if let Some(p) = pointer_pos {
            let d = from_screen(p) - from_screen(p - widgets::rated_delta(ui, &mv));
            for c in cs.iter_mut() {
                c[0] += d.x / size.0;
                c[1] += d.y / size.1;
            }
        }
    }
    for k in 0..4 {
        let sp = to_scr(cs[k]);
        let hr = ui.interact(
            egui::Rect::from_center_size(sp, egui::vec2(16.0, 16.0)),
            ui.id().with(("planar-corner", node, k)),
            egui::Sense::drag(),
        );
        ink.circle_stroke(sp, 4.0, 1.2, 0.9, hr.hovered());
        if hr.dragged() && !hr.drag_started() {
            if let Some(p) = pointer_pos {
                let d = from_screen(p) - from_screen(p - widgets::rated_delta(ui, &hr));
                cs[k][0] += d.x / size.0;
                cs[k][1] += d.y / size.1;
            }
        }
    }
    (cs != before).then_some(cs)
}

#[cfg(test)]
mod stabilize_tests {
    use super::*;
    use engine::graph::{CornerPinParams, NodeKind, UNIT_QUAD};
    use engine::planar::{h_apply, h_from_quad};

    /// The stabilise drive writes the track's pin into a CornerPin's FROM
    /// quad and leaves TO alone; the warp from -> to then carries the
    /// tracked region back onto where it sat on the reference frame.
    #[test]
    fn a_stabilise_pin_holds_the_region_still() {
        let (w, h) = (640.0f32, 360.0f32);
        let reference = [[0.30, 0.30], [0.70, 0.28], [0.72, 0.66], [0.28, 0.70]];
        // The region moved, turned and went into perspective.
        let moved = [[0.36, 0.33], [0.75, 0.36], [0.70, 0.74], [0.31, 0.69]];
        let (_, pin) = drive_of(&reference, &moved, (w, h));
        let mut kind = NodeKind::CornerPin(CornerPinParams::default());
        for (k, ch) in drive_channels(DriveTarget::Stabilize).iter().enumerate() {
            crate::anim::set(&mut kind, *ch, pin[k / 2][k % 2]);
        }
        let NodeKind::CornerPin(cp) = &kind else { unreachable!() };
        assert_eq!(cp.to, UNIT_QUAD, "the stabilise drive never touches TO");
        assert_eq!(cp.from, pin);
        let px = |q: [[f32; 2]; 4]| q.map(|p| [p[0] as f64 * w as f64, p[1] as f64 * h as f64]);
        // The warp the shader runs is frame -> effective_to; content at the
        // frame point p lands at (unit -> effective_to)(p).
        let shown = h_from_quad(&px(UNIT_QUAD), &px(cp.effective_to())).unwrap();
        for (r, m) in px(reference).iter().zip(px(moved).iter()) {
            let (x, y) = h_apply(&shown, m[0], m[1]);
            assert!((x - r[0]).hypot(y - r[1]) < 1e-2, "{m:?} landed at ({x}, {y}), wanted {r:?}");
        }
        // At the reference frame the pin is the frame: nothing moves.
        let (_, rest) = drive_of(&reference, &reference, (w, h));
        for k in 0..4 {
            assert!((rest[k][0] - UNIT_QUAD[k][0]).abs() < 1e-5 && (rest[k][1] - UNIT_QUAD[k][1]).abs() < 1e-5);
        }
    }

    #[test]
    fn the_pin_drives_still_reach_the_pin() {
        assert_eq!(drive_channels(DriveTarget::Transform)[0], Channel::Pin0X);
        assert_eq!(drive_channels(DriveTarget::CornerPin)[7], Channel::Pin3Y);
        assert_eq!(drive_channels(DriveTarget::Stabilize)[3], Channel::From1Y);
    }
}
