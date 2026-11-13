//! Point tracking for the Tracker node.
//!
//! An explicit BAKE, never live: tracking wants the whole sequence, and
//! decoding frames per playhead move would be exactly the kind of quiet slow
//! path this project refuses. The bake runs on a worker thread, streams a
//! position per frame back, and the app writes those as keyframes on the
//! node's TrackX/TrackY channels — so a patch cable from the tracker drives
//! anything with the motion of the plate.
//!
//! Two ways to find the point (`TrackMode`). PATTERN, the default, aligns
//! the square around the point into every frame with the planar tracker's
//! solve (`engine::pointtrack`): subpixel, follows a pattern that turns,
//! coasts through an occluder. BLOB is the thresholded luma centroid inside
//! a window that follows the previous result — no sub-pixel refinement and
//! no rotation, only a bright (or dark) blob on a plain ground.

use engine::decode::{ExrDecoder, ImageInfo};
use crate::anim::Channel;
use crate::editor::{EditorGraph, PSrc};
use engine::graph::{NodeKind, TrackMode};
use half::f16;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// One frame's result, streamed back as the bake runs.
pub enum TrackMsg {
    /// (frame index, every point's normalized position in order, whether
    /// EVERY point was actually found — a lost point coasts or holds).
    Point(u64, Vec<[f32; 2]>, bool),
    Done,
    Error(String),
}

pub struct TrackJob {
    rx: std::sync::mpsc::Receiver<TrackMsg>,
    cancel: Arc<AtomicBool>,
    /// First frame of the track: the playhead it was launched from, as a
    /// comp frame (tick).
    pub start: u64,
    /// The tick index 0 of the frame list plays at: results are keyed at
    /// `tick0 + index`, never at the index.
    tick0: i64,
    /// Frames this track covers — `start` to the end of the sequence.
    pub total: u64,
    pub done: u64,
    /// Frames where the blob was lost and the track coasted.
    pub lost: u64,
    pub error: Option<String>,
    pub finished: bool,
}

impl Drop for TrackJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl TrackJob {
    /// Drain whatever the worker has produced. Returns the points to key,
    /// by TICK.
    pub fn poll(&mut self) -> Vec<(u64, Vec<[f32; 2]>)> {
        let mut out = Vec::new();
        for msg in self.rx.try_iter() {
            match msg {
                TrackMsg::Point(i, p, found) => {
                    self.done += 1;
                    if !found {
                        self.lost += 1;
                    }
                    out.push(((self.tick0 + i as i64).max(0) as u64, p));
                }
                TrackMsg::Done => self.finished = true,
                TrackMsg::Error(e) => {
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
}

/// Point `i`'s position channels: point 0 is TrackX/TrackY (what a
/// single-point Tracker always keyed), the rest are indexed.
pub fn chans(i: usize) -> [Channel; 2] {
    if i == 0 {
        [Channel::TrackX, Channel::TrackY]
    } else {
        [Channel::TrackPtX(i as u16), Channel::TrackPtY(i as u16)]
    }
}

/// The solved corner pin's channels, in `drive_pins` order.
pub const PIN_CHANNELS: [Channel; 8] = [
    Channel::TrackPinX(0),
    Channel::TrackPinY(0),
    Channel::TrackPinX(1),
    Channel::TrackPinY(1),
    Channel::TrackPinX(2),
    Channel::TrackPinY(2),
    Channel::TrackPinX(3),
    Channel::TrackPinY(3),
];

/// The Tracker's inner graph: one node per point, each feeding Solve.
/// Zoom into the node to see it; what is wired into Solve (in port order)
/// is what drives the pin (`TrackerParams::solve_inputs`).
pub struct Points;
pub static POINTS: Points = Points;

/// The Solve node's id in the inner graph (points are 0..count).
pub const SOLVE: usize = 1 << 16;

impl crate::innards::Provider for Points {
    fn mini(&self, kind: &NodeKind) -> crate::innards::Mini {
        use crate::innards::{Mini, MiniEdge, MiniNode};
        let NodeKind::Tracker(t) = kind else { return Mini::default() };
        let mut nodes: Vec<MiniNode> = (0..t.count())
            .map(|i| MiniNode {
                id: i,
                at: t.node_at(i),
                label: (i + 1).to_string(),
                inputs: 0,
                output: true,
                item: true,
                removable: t.count() > 1,
            })
            .collect();
        let wired = t.solve_inputs();
        // One free port past the wired ones, so there is always somewhere
        // to plug the next point in.
        let ports = (wired.len() + 1).min(engine::graph::TRACK_POINTS_MAX);
        nodes.push(MiniNode {
            id: SOLVE,
            at: t.solve_node_at(),
            label: "S".into(),
            inputs: ports,
            output: false,
            item: false,
            removable: false,
        });
        let edges = wired.iter().enumerate().map(|(port, &from)| MiniEdge { from, to: SOLVE, port }).collect();
        Mini { nodes, edges }
    }

    fn move_to(&self, kind: &mut NodeKind, id: usize, at: [f32; 2]) {
        let NodeKind::Tracker(t) = kind else { return };
        if id == SOLVE {
            t.solve_at = Some(at);
        } else if id < t.count() {
            t.set_node_at(id, at);
        }
    }

    fn connect(&self, kind: &mut NodeKind, from: usize, to: usize, port: usize) -> bool {
        let NodeKind::Tracker(t) = kind else { return false };
        if to != SOLVE || from >= t.count() {
            return false;
        }
        t.wire_solve(from, port);
        true
    }

    fn disconnect(&self, kind: &mut NodeKind, to: usize, port: usize) {
        if let (NodeKind::Tracker(t), true) = (kind, to == SOLVE) {
            t.unwire_solve(port);
        }
    }

    fn can_add(&self, kind: &NodeKind) -> bool {
        matches!(kind, NodeKind::Tracker(t) if t.count() < engine::graph::TRACK_POINTS_MAX)
    }

    fn add(&self, graph: &mut EditorGraph, node: u64) -> Option<usize> {
        let n = graph.node_mut(node)?;
        let NodeKind::Tracker(t) = &mut n.kind else { return None };
        // A new point has no track yet, and the solve's reference covered
        // the old set: the pin rests until the next track.
        t.add_point()
    }

    fn remove(&self, graph: &mut EditorGraph, node: u64, i: usize) -> bool {
        let Some(n) = graph.node_mut(node) else { return false };
        let NodeKind::Tracker(t) = &mut n.kind else { return false };
        let count = t.count();
        if !t.remove_point(i) {
            return false;
        }
        // Slide the later points' curves down onto the freed index.
        for c in chans(i) {
            n.anim.remove(&c);
        }
        for k in i + 1..count {
            for (to, from) in chans(k - 1).into_iter().zip(chans(k)) {
                if let Some(c) = n.anim.remove(&from) {
                    n.anim.insert(to, c);
                }
            }
        }
        // ...and the cables reading them: one from the removed point goes,
        // later ones follow their point.
        let renum = |ch: Channel| -> Option<Option<Channel>> {
            let k = (0..count).find(|&k| chans(k).contains(&ch))?;
            let axis = chans(k).iter().position(|c| *c == ch)?;
            Some(if k == i { None } else if k > i { Some(chans(k - 1)[axis]) } else { Some(ch) })
        };
        graph.patch.links.retain_mut(|l| {
            if let PSrc::Chan(sn, sc) = &mut l.from {
                if *sn == node {
                    match renum(*sc) {
                        Some(None) => return false,
                        Some(Some(c)) => *sc = c,
                        None => {}
                    }
                }
            }
            true
        });
        true
    }
}

/// Track one frame: the luma-weighted centroid of everything past the
/// threshold inside the search window around `prev`. All positions are
/// NORMALIZED, so a track survives a resolution change.
///
/// Returns None when nothing in the window clears the threshold — the caller
/// holds the previous position rather than snapping to the window centre.
pub fn track_step(
    luma: &[f32],
    w: usize,
    h: usize,
    prev: [f32; 2],
    half: [f32; 2],
    threshold: f32,
    invert: bool,
) -> Option<[f32; 2]> {
    if w == 0 || h == 0 || luma.len() < w * h {
        return None;
    }
    let px = |v: f32, n: usize| (v * n as f32).round() as i64;
    let (cx, cy) = (px(prev[0], w), px(prev[1], h));
    let (hx, hy) = (px(half[0], w).max(1), px(half[1], h).max(1));
    let x0 = (cx - hx).clamp(0, w as i64 - 1) as usize;
    let x1 = (cx + hx).clamp(0, w as i64 - 1) as usize;
    let y0 = (cy - hy).clamp(0, h as i64 - 1) as usize;
    let y1 = (cy + hy).clamp(0, h as i64 - 1) as usize;
    let mut sum = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let l = luma[y * w + x];
            // Weight by how far past the threshold the pixel is, so the
            // centroid leans on the blob's core rather than its fringe.
            let weight = if invert { threshold - l } else { l - threshold };
            if weight > 0.0 {
                let weight = weight as f64;
                sum += weight;
                sx += weight * (x as f64 + 0.5);
                sy += weight * (y as f64 + 0.5);
            }
        }
    }
    if sum <= 1e-6 {
        return None;
    }
    Some([(sx / sum) as f32 / w as f32, (sy / sum) as f32 / h as f32])
}

/// Rec.709 luma of an RGBA-f16 frame, one f32 per pixel (row padding
/// dropped) — the buffer `track_step` walks.
pub fn luma_plane(src: &[f16], info: ImageInfo, row_stride: usize) -> Vec<f32> {
    let (w, h) = (info.width as usize, info.height as usize);
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let row = y * row_stride;
        for x in 0..w {
            let i = row + x * 4;
            if i + 2 < src.len() {
                let r = src[i].to_f32();
                let g = src[i + 1].to_f32();
                let b = src[i + 2].to_f32();
                out[y * w + x] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            }
        }
    }
    out
}

/// Kick a track over `frames`, starting from the node's authored region at
/// INDEX `start` of the list and running to its end; index 0 plays at comp
/// frame `tick0`.
///
/// `start` is the PLAYHEAD: the region you drew sits on the feature in the
/// frame you were looking at, so that is the frame the track is seeded
/// from. Frames before it are left unkeyed and the curve holds the first
/// key, rather than the track beginning at frame 0 where the region was
/// authored against nothing.
pub fn spawn(
    decoder: Arc<dyn ExrDecoder>,
    frames: Vec<PathBuf>,
    info: ImageInfo,
    start: u64,
    tick0: i64,
    params: engine::graph::TrackerParams,
) -> TrackJob {
    let (tx, rx) = std::sync::mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let start = start.min(frames.len().saturating_sub(1) as u64);
    let total = frames.len() as u64 - start;
    let stop = cancel.clone();
    crate::bg::spawn(move || {
        let stride = info.row_stride_f16();
        let mut buf = vec![f16::ZERO; stride * info.height as usize];
        let (w, h) = (info.width as usize, info.height as usize);
        let points: Vec<engine::graph::TrackPoint> =
            (0..params.count()).filter_map(|i| params.point(i)).collect();
        // Blob: each window follows its own last result, pattern size plus
        // the search radius. Pattern: one aligner per point, seeded on the
        // first frame.
        let mut prev: Vec<[f32; 2]> = points.iter().map(|p| p.center).collect();
        let mut pattern: Vec<engine::pointtrack::PointTracker> = Vec::new();
        for (i, path) in frames.iter().enumerate().skip(start as usize) {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            if let Err(e) = decoder.decode_into(path, info, stride, &mut buf) {
                let _ = tx.send(TrackMsg::Error(format!("{}: {e:#}", path.display())));
                return;
            }
            let mut all_found = true;
            match params.mode {
                TrackMode::Blob => {
                    let luma = luma_plane(&buf, info, stride);
                    for (k, p) in points.iter().enumerate() {
                        let half = [p.size[0] + p.search, p.size[1] + p.search];
                        match track_step(&luma, w, h, prev[k], half, params.threshold, params.invert) {
                            Some(q) => prev[k] = q,
                            None => all_found = false,
                        }
                    }
                }
                TrackMode::Pattern => {
                    let plane = engine::planar::Plane::from_rgba_f16(&buf, w, h, stride);
                    if pattern.is_empty() {
                        // The seed frame IS the pattern: its answer is each
                        // point as drawn.
                        pattern = points
                            .iter()
                            .map(|p| {
                                engine::pointtrack::PointTracker::new(
                                    plane.clone(),
                                    [p.center[0] as f64 * w as f64, p.center[1] as f64 * h as f64],
                                    [p.size[0] as f64 * w as f64, p.size[1] as f64 * h as f64],
                                    engine::pointtrack::PointOpts {
                                        model: params.model,
                                        search: p.search as f64 * w.max(h) as f64,
                                        ..Default::default()
                                    },
                                )
                            })
                            .collect();
                    } else {
                        // One smoothed pyramid for the frame, every point
                        // aligned on it.
                        let pyr = pattern[0].prepare(plane);
                        for (k, tr) in pattern.iter_mut().enumerate() {
                            let r = tr.step_prepared(&pyr);
                            prev[k] = [(r.pos[0] / w as f64) as f32, (r.pos[1] / h as f64) as f32];
                            all_found &= !r.lost;
                        }
                    }
                }
            }
            if tx.send(TrackMsg::Point(i as u64, prev.clone(), all_found)).is_err() {
                return; // the node went away
            }
        }
        let _ = tx.send(TrackMsg::Done);
    });
    let start_tick = (tick0 + start as i64).max(0) as u64;
    TrackJob { rx, cancel, start: start_tick, tick0, total, done: 0, lost: 0, error: None, finished: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame with one bright disc at (cx, cy) in pixels.
    fn blob(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<f32> {
        let mut v = vec![0.1f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
                if dx * dx + dy * dy <= r * r {
                    v[y * w + x] = 0.9;
                }
            }
        }
        v
    }

    #[test]
    fn centroid_finds_the_blob_and_follows_it() {
        let (w, h) = (64usize, 64usize);
        let f = blob(w, h, 20.0, 30.0, 5.0);
        let p = track_step(&f, w, h, [0.5, 0.5], [0.5, 0.5], 0.5, false).unwrap();
        // Normalized centre of the disc, within a fraction of a pixel.
        assert!((p[0] - 20.0 / 64.0).abs() < 0.01, "{p:?}");
        assert!((p[1] - 30.0 / 64.0).abs() < 0.01, "{p:?}");

        // A small window centred on the last result keeps up with motion...
        let moved = blob(w, h, 26.0, 30.0, 5.0);
        let q = track_step(&moved, w, h, p, [0.15, 0.15], 0.5, false).unwrap();
        assert!((q[0] - 26.0 / 64.0).abs() < 0.01, "{q:?}");

        // ...and a blob that jumps outside the window is reported LOST
        // rather than silently snapping to whatever is nearby.
        let jumped = blob(w, h, 60.0, 60.0, 5.0);
        assert!(track_step(&jumped, w, h, q, [0.05, 0.05], 0.5, false).is_none());
    }

    #[test]
    fn inverted_tracks_dark_blobs() {
        let (w, h) = (32usize, 32usize);
        // A dark disc on a light field: the same disc, luma flipped.
        let bright = blob(w, h, 10.0, 12.0, 4.0);
        let dark: Vec<f32> = bright.iter().map(|v| 1.0 - v).collect();
        let p = track_step(&dark, w, h, [0.5, 0.5], [0.5, 0.5], 0.5, true).unwrap();
        assert!((p[0] - 10.0 / 32.0).abs() < 0.02, "{p:?}");
        assert!((p[1] - 12.0 / 32.0).abs() < 0.02, "{p:?}");
        // Without invert the bright BACKGROUND is what clears the threshold,
        // so the centroid lands near the middle of the frame — nowhere near
        // the disc. (Tracking the wrong thing, correctly.)
        let q = track_step(&dark, w, h, [0.5, 0.5], [0.5, 0.5], 0.5, false).unwrap();
        assert!((q[0] - 0.5).abs() < 0.1 && (q[1] - 0.5).abs() < 0.1, "{q:?}");
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert!(track_step(&[], 0, 0, [0.5, 0.5], [0.1, 0.1], 0.5, false).is_none());
        // A short buffer is refused, not read past.
        assert!(track_step(&[0.9; 4], 8, 8, [0.5, 0.5], [0.1, 0.1], 0.5, false).is_none());
        // A position far outside the image clamps into it (no out-of-bounds
        // index); with the blob outside the clamped window it reports LOST.
        let f = blob(16, 16, 2.0, 2.0, 2.0);
        assert!(track_step(&f, 16, 16, [-5.0, 9.0], [0.9, 0.9], 0.5, false).is_none());
        // Nudged back over the blob, the same clamping still finds it.
        let p = track_step(&f, 16, 16, [-0.1, 0.1], [0.9, 0.9], 0.5, false).unwrap();
        assert!(p[0] >= 0.0 && p[0] <= 1.0 && p[1] >= 0.0 && p[1] <= 1.0, "{p:?}");
    }

    #[test]
    fn removing_a_point_slides_the_later_curves_and_cables_down() {
        use crate::innards::Provider;
        use crate::editor::{PDst, PatchLink};
        let mut g = EditorGraph::default();
        let id = g.add(NodeKind::Tracker(Default::default()), egui::Pos2::ZERO);
        let other = g.add(NodeKind::Tracker(Default::default()), egui::Pos2::ZERO);
        for _ in 0..3 {
            POINTS.add(&mut g, id);
        }
        // A curve per point, told apart by value; a cable from points 1 and 3.
        {
            let n = g.node_mut(id).unwrap();
            for i in 0..4 {
                n.anim.entry(chans(i)[0]).or_default().set_key(0, i as f32);
            }
        }
        for i in [1, 3] {
            g.patch.links.push(PatchLink {
                from: PSrc::Chan(id, chans(i)[0]),
                to: PDst::Chan(other, Channel::TrackX),
            });
        }
        assert!(POINTS.remove(&mut g, id, 1));
        let n = g.node(id).unwrap();
        let v = |i: usize| n.anim.get(&chans(i)[0]).and_then(|c| c.eval_at(0.0));
        assert_eq!((v(0), v(1), v(2)), (Some(0.0), Some(2.0), Some(3.0)), "curves followed their points");
        assert!(v(3).is_none(), "nothing left at the old last index");
        let from: Vec<PSrc> = g.patch.links.iter().map(|l| l.from).collect();
        assert_eq!(from, vec![PSrc::Chan(id, chans(2)[0])], "point 1's cable went, point 3's followed it");
        // Point 0 can go too (point 1 becomes TrackX), the last one cannot.
        assert!(POINTS.remove(&mut g, id, 0));
        assert_eq!(g.node(id).unwrap().anim.get(&Channel::TrackX).and_then(|c| c.eval_at(0.0)), Some(2.0));
        assert!(POINTS.remove(&mut g, id, 0));
        assert!(!POINTS.remove(&mut g, id, 0), "a Tracker keeps one point");
    }

    #[test]
    fn luma_plane_drops_row_padding() {
        let info = ImageInfo { width: 2, height: 2 };
        let stride = info.row_stride_f16();
        let mut src = vec![f16::ZERO; stride * 2];
        // Pixel (1, 1) is pure green: luma 0.7152.
        src[stride + 4 + 1] = f16::from_f32(1.0);
        let l = luma_plane(&src, info, stride);
        assert_eq!(l.len(), 4);
        assert!((l[3] - 0.7152).abs() < 1e-3, "{l:?}");
        assert_eq!(l[0], 0.0);
    }
}
