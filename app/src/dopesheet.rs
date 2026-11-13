//! Dope sheet: the TIMING view of the same keys the curve editor shapes.
//! One row per animated channel, grouped under its node, with a summary row
//! on top; keys are diamonds on a frame ruler and values never show. It is
//! Blender's Dope Sheet, and so are its hands (manifesto rule 11):
//!
//! - click selects, shift-click toggles, drag on bare rows box-selects
//!   (shift adds, ctrl subtracts); clicking a node or summary diamond takes
//!   every key under it on that frame
//! - drag a key to slide the selection in whole frames; G grabs, S scales
//!   about the playhead, shift+D duplicates and grabs — LMB/Enter confirms,
//!   RMB/Esc cancels
//! - A selects all, alt+A none, ctrl+I inverts, X/Delete deletes,
//!   ⌘C/⌘V copies keys and pastes them at the playhead
//! - the ruler scrubs; wheel zooms time, trackpad/ctrl/shift-wheel pan,
//!   MMB (alt+LMB) pans, R/Home frames
//!
//! A key that lands on another key of the same channel REPLACES it, as in
//! Blender. Locked channels are drawn in the lock colour and take no edits.
//! Every edit is a plain mutation of `ENode::anim`; the app's per-frame graph
//! diff makes it an undo step like any other key edit.

use crate::anim::{frame_id, id_frame, AnimCurve, Channel, FrameId, Interp, Key};
use crate::chanview::ChannelView;
use crate::editor::EditorGraph;
use crate::theme;
use egui::{Color32, Pos2, Rect, Stroke};
use std::collections::{BTreeSet, HashMap, HashSet};

/// A key's identity: node, channel, frame (keys can sit between frames, and
/// an f64 does not hash — see `anim::FrameId`).
type KeyId = (u64, Channel, FrameId);
/// Curves as they were when a gesture began.
type Snapshot = HashMap<(u64, Channel), AnimCurve>;

const ROW_H: f32 = 18.0;
const RULER_H: f32 = 20.0;
const LIST_W: f32 = 190.0;
/// Horizontal pick radius for a diamond, in points.
const HIT: f32 = 6.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowKind {
    Summary,
    Node(u64),
    Chan(u64, Channel),
}

struct Row {
    kind: RowKind,
    label: String,
    locked: bool,
    /// Sorted, unique frames that carry a key in this row.
    frames: Vec<f64>,
}

/// A key gesture. Mouse drags and the modal G/S/shift+D share it; `modal`
/// says whether the pointer's MOTION drives it (no button held) and a click
/// ends it.
enum Op {
    Pan,
    Scrub,
    Marquee { start: Pos2 },
    Move { snapshot: Snapshot, grabbed: Vec<(u64, Channel, Key)>, origin_x: f32, dup: bool, modal: bool },
    Scale { snapshot: Snapshot, grabbed: Vec<(u64, Channel, Key)>, origin_x: f32, pivot: f32, modal: bool },
}

pub struct DopeSheet {
    /// Frame at the left edge of the key area, and points per frame.
    x_off: f32,
    x_scale: f32,
    /// Rows scrolled off the top, in points.
    y_scroll: f32,
    op: Option<Op>,
    selection: HashSet<KeyId>,
    /// Collapsed node groups.
    collapsed: HashSet<u64>,
    /// Keys copied with ⌘C, frames relative to the earliest.
    clip: Vec<(u64, Channel, Key)>,
    framed: bool,
    /// Last painted key area (self-tests aim at it).
    pub last_rect: Rect,
    /// What the last op did, for the status line.
    note: String,
}

impl Default for DopeSheet {
    fn default() -> Self {
        Self {
            x_off: 0.0,
            x_scale: 8.0,
            y_scroll: 0.0,
            op: None,
            selection: HashSet::new(),
            collapsed: HashSet::new(),
            clip: Vec::new(),
            framed: false,
            last_rect: Rect::NOTHING,
            note: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Pure key edits (unit tested below). Every gesture restores its snapshot and
// re-applies from scratch each frame, so a wandering drag never drifts.
// ---------------------------------------------------------------------------

fn restore(graph: &mut EditorGraph, snapshot: &Snapshot) {
    for ((n, ch), curve) in snapshot {
        if let Some(nd) = graph.nodes.get_mut(n) {
            nd.anim.insert(*ch, curve.clone());
        }
    }
}

/// Lift `grabbed` off their curves (unless duplicating) and set them down at
/// `map(frame)`, replacing whatever key sat there. Returns the new selection.
fn place(
    graph: &mut EditorGraph,
    grabbed: &[(u64, Channel, Key)],
    dup: bool,
    map: impl Fn(f64) -> f64,
) -> HashSet<KeyId> {
    if !dup {
        for (n, ch, k) in grabbed {
            if let Some(c) = graph.nodes.get_mut(n).and_then(|nd| nd.anim.get_mut(ch)) {
                c.remove_key(k.frame);
            }
        }
    }
    let mut sel = HashSet::new();
    for (n, ch, k) in grabbed {
        let Some(c) = graph.nodes.get_mut(n).and_then(|nd| nd.anim.get_mut(ch)) else {
            continue;
        };
        let f = map(k.frame);
        let key = Key { frame: f, ..*k };
        match c.keys.binary_search_by(|k| k.frame.total_cmp(&f)) {
            Ok(i) => c.keys[i] = key,
            Err(i) => c.keys.insert(i, key),
        }
        sel.insert((*n, *ch, frame_id(f)));
    }
    sel
}

/// Ruler step: the 1/2/5 x 10^n frames nearest `target`.
fn nice_step(target: f32) -> f32 {
    if !target.is_finite() || target <= 0.0 {
        return 1.0;
    }
    let mag = 10f32.powf(target.log10().floor());
    let n = target / mag;
    mag * if n < 1.5 { 1.0 } else if n < 3.5 { 2.0 } else if n < 7.5 { 5.0 } else { 10.0 }
}

/// Whole-frame steps; a sub-frame key keeps its fraction.
fn shift(f: f64, df: i64) -> f64 {
    (f + df as f64).max(0.0)
}

fn move_keys(
    graph: &mut EditorGraph,
    snapshot: &Snapshot,
    grabbed: &[(u64, Channel, Key)],
    df: i64,
    dup: bool,
) -> HashSet<KeyId> {
    restore(graph, snapshot);
    place(graph, grabbed, dup, |f| shift(f, df))
}

fn scale_keys(
    graph: &mut EditorGraph,
    snapshot: &Snapshot,
    grabbed: &[(u64, Channel, Key)],
    pivot: f32,
    s: f32,
) -> HashSet<KeyId> {
    restore(graph, snapshot);
    place(graph, grabbed, false, |f| (pivot as f64 + (f - pivot as f64) * s as f64).round().max(0.0))
}

/// Remove keys; a curve left empty goes too (unless it is locked — the lock
/// lives on the curve and must survive).
fn delete_keys(graph: &mut EditorGraph, keys: impl IntoIterator<Item = KeyId>) {
    for (n, ch, f) in keys {
        let Some(nd) = graph.nodes.get_mut(&n) else { continue };
        if let Some(c) = nd.anim.get_mut(&ch) {
            c.remove_key(id_frame(f));
            if c.keys.is_empty() && !c.locked {
                nd.anim.remove(&ch);
            }
        }
    }
}

/// Paste copied keys so the earliest lands on `t`. Returns what was pasted.
pub(crate) fn paste_keys(graph: &mut EditorGraph, clip: &[(u64, Channel, Key)], t: u64) -> HashSet<KeyId> {
    let mut sel = HashSet::new();
    for (n, ch, k) in clip {
        let Some(nd) = graph.nodes.get_mut(n) else { continue };
        let c = nd.anim.entry(*ch).or_default();
        if c.locked {
            continue;
        }
        let f = t as f64 + k.frame;
        let key = Key { frame: f, ..*k };
        match c.keys.binary_search_by(|k| k.frame.total_cmp(&f)) {
            Ok(i) => c.keys[i] = key,
            Err(i) => c.keys.insert(i, key),
        }
        sel.insert((*n, *ch, frame_id(f)));
    }
    sel
}

/// Lock-safe snapshot of the keys a gesture moves.
fn grab(graph: &EditorGraph, sel: &HashSet<KeyId>) -> (Snapshot, Vec<(u64, Channel, Key)>) {
    let mut snapshot = Snapshot::new();
    let mut grabbed = Vec::new();
    for (n, ch, f) in sel.iter().copied() {
        let Some(c) = graph.nodes.get(&n).and_then(|nd| nd.anim.get(&ch)) else { continue };
        if c.locked {
            continue;
        }
        let Some(k) = c.keys.iter().find(|k| k.frame == id_frame(f)) else { continue };
        snapshot.entry((n, ch)).or_insert_with(|| c.clone());
        grabbed.push((n, ch, *k));
    }
    (snapshot, grabbed)
}

impl DopeSheet {
    fn x_of(&self, rect: &Rect, f: f32) -> f32 {
        rect.min.x + (f - self.x_off) * self.x_scale
    }
    fn frame_at(&self, rect: &Rect, x: f32) -> f32 {
        self.x_off + (x - rect.min.x) / self.x_scale
    }

    /// Screen x of `frame` in the last painted view (self-tests).
    #[allow(dead_code)]
    pub fn key_x(&self, frame: f32) -> f32 {
        self.x_of(&self.last_rect, frame)
    }

    fn frame_all(&mut self, rect: &Rect, rows: &[Row]) {
        let lo = rows.iter().filter_map(|r| r.frames.first().copied()).reduce(f64::min);
        let hi = rows.iter().filter_map(|r| r.frames.last().copied()).reduce(f64::max);
        let (Some(lo), Some(hi)) = (lo, hi) else { return };
        let span = (hi - lo).max(10.0) as f32;
        self.x_scale = (rect.width() * 0.86 / span).clamp(0.05, 200.0);
        self.x_off = lo as f32 - rect.width() * 0.07 / self.x_scale;
    }

    /// Keys a row stands for at `frame` (a summary/node row stands for all
    /// the channel keys beneath it). Locked channels contribute nothing.
    fn keys_under(&self, graph: &EditorGraph, chans: &[(u64, Channel)], kind: RowKind, frame: f64) -> Vec<KeyId> {
        chans
            .iter()
            .filter(|(n, ch)| match kind {
                RowKind::Summary => true,
                RowKind::Node(id) => *n == id,
                RowKind::Chan(id, c) => *n == id && *ch == c,
            })
            .filter(|(n, ch)| {
                graph
                    .nodes
                    .get(n)
                    .and_then(|nd| nd.anim.get(ch))
                    .is_some_and(|c| !c.locked && c.has_key(frame))
            })
            .map(|(n, ch)| (*n, *ch, frame_id(frame)))
            .collect()
    }

    /// Draw the sheet and edit keys in place. `t` is the playhead; the
    /// return is a frame to seek to (ruler scrub).
    /// `view` is the curve editor's channel view: the rows are EXACTLY the
    /// channels it shows, in its order.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        graph: &mut EditorGraph,
        view: &mut ChannelView,
        sel: &[u64],
        t: u64,
    ) -> Option<u64> {
        let mut seek = None;

        // --- which channels: the curve editor's visible ones ---
        view.prune(graph);
        let listed = view.listed(graph, sel, |n, ch| crate::curves::track_label(graph, n, ch));
        let n_hidden = listed.len() - view.visible(&listed).len();
        let mut groups: Vec<(u64, String, Vec<(Channel, String)>)> = Vec::new();
        for (id, ch) in view.visible(&listed) {
            let Some(n) = graph.nodes.get(&id) else { continue };
            let l = crate::curves::ch_label_for(&n.kind, ch).into_owned();
            match groups.last_mut() {
                Some(g) if g.0 == id => g.2.push((ch, l)),
                _ => groups.push((id, crate::editor::node_label(&n.kind), vec![(ch, l)])),
            }
        }
        // Forget selected keys that no longer exist (undo, another editor).
        self.selection.retain(|(n, ch, f)| {
            graph.nodes.get(n).and_then(|nd| nd.anim.get(ch)).is_some_and(|c| c.has_key(id_frame(*f)))
        });

        let all_chans: Vec<(u64, Channel)> =
            groups.iter().flat_map(|(id, _, cs)| cs.iter().map(|(c, _)| (*id, *c))).collect();
        let frames_of = |pairs: &mut dyn Iterator<Item = (u64, Channel)>| -> Vec<f64> {
            let mut s = BTreeSet::new();
            for (n, ch) in pairs {
                if let Some(c) = graph.nodes.get(&n).and_then(|nd| nd.anim.get(&ch)) {
                    s.extend(c.keys.iter().map(|k| frame_id(k.frame)));
                }
            }
            s.into_iter().map(id_frame).collect()
        };
        let mut rows: Vec<Row> = Vec::new();
        if !groups.is_empty() {
            rows.push(Row {
                kind: RowKind::Summary,
                label: "summary".into(),
                locked: false,
                frames: frames_of(&mut all_chans.iter().copied()),
            });
        }
        for (id, label, chans) in &groups {
            rows.push(Row {
                kind: RowKind::Node(*id),
                label: label.clone(),
                locked: false,
                frames: frames_of(&mut chans.iter().map(|(c, _)| (*id, *c))),
            });
            if self.collapsed.contains(id) {
                continue;
            }
            for (ch, l) in chans {
                let c = &graph.nodes[id].anim[ch];
                rows.push(Row {
                    kind: RowKind::Chan(*id, *ch),
                    label: l.clone(),
                    locked: c.locked,
                    frames: c.keys.iter().map(|k| k.frame).collect(),
                });
            }
        }

        // --- header: interpolation of the selection + frame ---
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("dope sheet").strong());
            view.sets_menu(ui, &listed);
            if n_hidden > 0
                && ui
                    .add(egui::Button::new(egui::RichText::new(format!("show {n_hidden} hidden")).small()))
                    .on_hover_text("channels hidden here are hidden in the curve editor too")
                    .clicked()
            {
                view.hidden.clear();
            }
            if rows.is_empty() {
                ui.weak(if !listed.is_empty() {
                    "every channel is hidden"
                } else if sel.is_empty() {
                    "nothing animated yet — key a parameter to see it here"
                } else {
                    "no animated channels on the selection"
                });
                return;
            }
            ui.separator();
            let has_sel = !self.selection.is_empty();
            let current = {
                let mut it = self.selection.iter().filter_map(|(n, ch, f)| {
                    graph.nodes.get(n)?.anim.get(ch)?.keys.iter().find(|k| k.frame == id_frame(*f)).map(|k| k.interp)
                });
                match it.next() {
                    Some(m) if it.all(|o| o == m) => Some(m),
                    _ => None,
                }
            };
            for (name, mode) in [
                ("hold", Interp::Hold),
                ("linear", Interp::Linear),
                ("smooth", Interp::Smooth),
                ("bezier", Interp::Bezier),
            ] {
                let r = ui
                    .add_enabled(
                        has_sel,
                        egui::Button::new(egui::RichText::new(name).small()).selected(current == Some(mode)),
                    )
                    .on_hover_text(if has_sel {
                        "set the selected keys' interpolation, both sides\n\
                         shift-click: only the segment leaving each key"
                    } else {
                        "select keys first"
                    });
                if r.clicked() {
                    // Both sides of each key by default; shift sets only
                    // the segment leaving it (the curve editor's rule).
                    let out_only = ui.input(|i| i.modifiers.shift);
                    for (n, ch, f) in self.selection.iter().copied() {
                        if let Some(c) = graph.nodes.get_mut(&n).and_then(|nd| nd.anim.get_mut(&ch)) {
                            if !c.locked {
                                if out_only {
                                    c.set_interp(id_frame(f), mode);
                                } else {
                                    c.set_interp_both(id_frame(f), mode);
                                }
                            }
                        }
                    }
                }
            }
            ui.separator();
            if ui
                .add(egui::Button::new(egui::RichText::new("frame").small()))
                .on_hover_text("frame every key (R / Home)")
                .clicked()
            {
                self.framed = false;
            }
        });

        let full = ui.available_rect_before_wrap();
        if full.width() < LIST_W + 48.0 || full.height() < RULER_H + ROW_H {
            return None;
        }
        let resp = ui.allocate_rect(full, egui::Sense::click_and_drag());
        let painter = ui.painter_at(full);
        let list = Rect::from_min_max(full.min, Pos2::new(full.min.x + LIST_W, full.max.y));
        let ruler = Rect::from_min_max(Pos2::new(list.max.x, full.min.y), Pos2::new(full.max.x, full.min.y + RULER_H));
        let area = Rect::from_min_max(Pos2::new(list.max.x, ruler.max.y), full.max);
        self.last_rect = area;
        painter.rect_filled(full, 0.0, theme::well());
        painter.rect_filled(list, 0.0, theme::bg());
        painter.rect_filled(ruler, 0.0, theme::raised_fill());
        if rows.is_empty() {
            return None;
        }
        if !self.framed {
            self.framed = true;
            self.frame_all(&area, &rows);
        }

        let content_h = rows.len() as f32 * ROW_H;
        let max_scroll = (content_h - (area.height() - ROW_H)).max(0.0);
        self.y_scroll = self.y_scroll.clamp(0.0, max_scroll);
        let row_top = |i: usize, s: &Self| area.min.y + i as f32 * ROW_H - s.y_scroll;

        // --- navigation: time zooms, rows scroll ---
        let pointer = resp.hover_pos().or_else(|| ui.ctx().pointer_latest_pos());
        let mods = ui.input(|i| i.modifiers);
        let over = ui.rect_contains_pointer(full);
        if over {
            let w = crate::nav::wheel_nav(ui);
            self.x_off -= w.pan.x / self.x_scale;
            self.y_scroll -= w.pan.y;
            if (w.zoom - 1.0).abs() > 1e-4 {
                if let Some(c) = pointer {
                    let fr = self.frame_at(&area, c.x);
                    self.x_scale = (self.x_scale * w.zoom).clamp(0.05, 200.0);
                    self.x_off = fr - (c.x - area.min.x) / self.x_scale;
                }
            }
        }
        let typing = ui.ctx().egui_wants_keyboard_input() || crate::keymap::capturing();
        if over
            && !typing
            && ui.input(|i| crate::keymap::pressed(i, crate::keymap::Act::CurvesFrame))
        {
            self.frame_all(&area, &rows);
        }

        // --- hit testing ---
        let row_at = |y: f32, s: &Self| -> Option<usize> {
            if y < area.min.y || y > area.max.y {
                return None;
            }
            let i = ((y - area.min.y + s.y_scroll) / ROW_H).floor();
            (i >= 0.0 && (i as usize) < rows.len()).then_some(i as usize)
        };
        let key_at = |p: Pos2, s: &Self| -> Option<(usize, f64)> {
            if !area.contains(p) {
                return None;
            }
            let i = row_at(p.y, s)?;
            if rows[i].locked {
                return None;
            }
            rows[i]
                .frames
                .iter()
                .map(|f| (*f, (s.x_of(&area, *f as f32) - p.x).abs()))
                .filter(|(_, d)| *d <= HIT)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(f, _)| (i, f))
        };
        let pick = |s: &Self, g: &EditorGraph, hit: (usize, f64)| s.keys_under(g, &all_chans, rows[hit.0].kind, hit.1);

        // --- modal ops (G / S / shift+D): pointer motion drives, click ends ---
        let modal = matches!(self.op, Some(Op::Move { modal: true, .. } | Op::Scale { modal: true, .. }));
        if modal {
            let confirm = resp.clicked() || ui.input(|i| i.key_pressed(egui::Key::Enter));
            let cancel = resp.secondary_clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape));
            if cancel {
                if let Some(Op::Move { snapshot, grabbed, .. } | Op::Scale { snapshot, grabbed, .. }) = self.op.take() {
                    // Back to exactly how it was; a cancelled duplicate
                    // leaves the originals selected.
                    restore(graph, &snapshot);
                    self.selection = grabbed.iter().map(|(n, ch, k)| (*n, *ch, frame_id(k.frame))).collect();
                }
                self.note = "cancelled".into();
            } else if confirm {
                self.op = None;
            }
        }

        // --- keyboard (only while the pointer is over the sheet) ---
        if over && !typing && self.op.is_none() {
            let (a, alt_a, inv, del, g, s, dup) = ui.input_mut(|i| {
                let alt_a = i.consume_key(egui::Modifiers::ALT, egui::Key::A);
                (
                    i.consume_key(egui::Modifiers::NONE, egui::Key::A),
                    alt_a,
                    i.consume_key(egui::Modifiers::COMMAND, egui::Key::I),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::X)
                        || i.consume_key(egui::Modifiers::NONE, egui::Key::Delete)
                        || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::G),
                    i.consume_key(egui::Modifiers::NONE, egui::Key::S),
                    i.consume_key(egui::Modifiers::SHIFT, egui::Key::D),
                )
            });
            let every = || -> HashSet<KeyId> {
                all_chans
                    .iter()
                    .filter_map(|(n, ch)| Some((*n, *ch, graph.nodes.get(n)?.anim.get(ch)?)))
                    .filter(|(.., c)| !c.locked)
                    .flat_map(|(n, ch, c)| c.keys.iter().map(move |k| (n, ch, frame_id(k.frame))))
                    .collect()
            };
            if a {
                self.selection = every();
            }
            if alt_a {
                self.selection.clear();
            }
            if inv {
                self.selection = every().difference(&self.selection).copied().collect();
            }
            if del && !self.selection.is_empty() {
                self.note = format!("deleted {} key(s)", self.selection.len());
                delete_keys(graph, self.selection.drain());
            }
            let x = pointer.map_or(area.center().x, |p| p.x);
            if (g || dup) && !self.selection.is_empty() {
                let (snapshot, grabbed) = grab(graph, &self.selection);
                self.op = Some(Op::Move { snapshot, grabbed, origin_x: x, dup, modal: true });
            } else if s && !self.selection.is_empty() {
                let (snapshot, grabbed) = grab(graph, &self.selection);
                self.op = Some(Op::Scale { snapshot, grabbed, origin_x: x, pivot: t as f32, modal: true });
            }
            // ⌘C / ⌘V arrive as egui's own events, never key presses.
            let (copy, paste) = ui.input(|i| {
                i.events.iter().fold((false, false), |acc, e| match e {
                    egui::Event::Copy => (true, acc.1),
                    egui::Event::Paste(_) => (acc.0, true),
                    _ => acc,
                })
            });
            if copy && !self.selection.is_empty() {
                let (_, mut grabbed) = grab(graph, &self.selection);
                let f0 = grabbed.iter().map(|(.., k)| k.frame).reduce(f64::min).unwrap_or(0.0);
                for (.., k) in &mut grabbed {
                    k.frame -= f0;
                }
                self.note = format!("copied {} key(s)", grabbed.len());
                self.clip = grabbed;
            }
            if paste && !self.clip.is_empty() {
                self.selection = paste_keys(graph, &self.clip, t);
                self.note = format!("pasted {} key(s) at {t}", self.selection.len());
            }
        }

        // --- mouse: start a gesture ---
        if resp.drag_started_by(egui::PointerButton::Middle) && self.op.is_none() {
            self.op = Some(Op::Pan);
        }
        if resp.drag_started_by(egui::PointerButton::Primary) && self.op.is_none() {
            let origin = ui.input(|i| i.pointer.press_origin()).or(pointer);
            if let Some(p) = origin {
                self.op = if mods.alt {
                    Some(Op::Pan) // Blender's emulated MMB
                } else if ruler.contains(p) {
                    Some(Op::Scrub)
                } else if list.contains(p) {
                    None
                } else if let Some(hit) = key_at(p, self) {
                    let under = pick(self, graph, hit);
                    if !under.iter().any(|k| self.selection.contains(k)) {
                        if !mods.shift {
                            self.selection.clear();
                        }
                        self.selection.extend(under);
                    }
                    let (snapshot, grabbed) = grab(graph, &self.selection);
                    Some(Op::Move { snapshot, grabbed, origin_x: p.x, dup: false, modal: false })
                } else {
                    Some(Op::Marquee { start: p })
                };
            }
        }

        // --- mouse: run the gesture ---
        let mut marquee = None;
        let dragging = resp.dragged();
        match &self.op {
            Some(Op::Pan) if dragging => {
                let d = resp.drag_delta();
                self.x_off -= d.x / self.x_scale;
                self.y_scroll -= d.y;
            }
            Some(Op::Scrub) if dragging || resp.drag_started() => {
                if let Some(p) = pointer {
                    seek = Some(self.frame_at(&area, p.x).round().max(0.0) as u64);
                }
            }
            Some(Op::Marquee { start }) if dragging => {
                if let Some(p) = pointer {
                    marquee = Some(Rect::from_two_pos(*start, p));
                }
            }
            Some(Op::Move { snapshot, grabbed, origin_x, dup, modal }) if dragging || *modal => {
                if let Some(p) = pointer {
                    let df = (self.frame_at(&area, p.x) - self.frame_at(&area, *origin_x)).round() as i64;
                    self.selection = move_keys(graph, snapshot, grabbed, df, *dup);
                    self.note = format!("{} {:+} frame(s)", if *dup { "duplicate" } else { "move" }, df);
                }
            }
            Some(Op::Scale { snapshot, grabbed, origin_x, pivot, modal }) if dragging || *modal => {
                if let Some(p) = pointer {
                    let px = self.x_of(&area, *pivot);
                    let was = *origin_x - px;
                    let s = if was.abs() > 1.0 { ((p.x - px) / was).max(0.0) } else { 1.0 };
                    self.selection = scale_keys(graph, snapshot, grabbed, *pivot, s);
                    self.note = format!("scale ×{s:.2} about {}", *pivot as i64);
                }
            }
            _ => {}
        }
        if resp.drag_stopped() && !modal {
            if let Some(Op::Marquee { start }) = self.op.take() {
                if let Some(p) = pointer {
                    let r = Rect::from_two_pos(start, p);
                    if !mods.shift && !mods.command {
                        self.selection.clear();
                    }
                    for (i, row) in rows.iter().enumerate() {
                        let y = row_top(i, self) + ROW_H * 0.5;
                        if !(r.min.y..=r.max.y).contains(&y) {
                            continue;
                        }
                        for f in &row.frames {
                            if (r.min.x..=r.max.x).contains(&self.x_of(&area, *f as f32)) {
                                for k in pick(self, graph, (i, *f)) {
                                    if mods.command {
                                        self.selection.remove(&k);
                                    } else {
                                        self.selection.insert(k);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            self.op = None;
        }

        // Right-click a channel's name: lock / unlock it, as in the curve list.
        if !modal && resp.secondary_clicked() {
            if let Some(p) = pointer.filter(|p| list.contains(*p)) {
                if let Some(RowKind::Chan(n, ch)) = row_at(p.y, self).map(|i| rows[i].kind) {
                    if let Some(nd) = graph.nodes.get_mut(&n) {
                        let locked = crate::anim::is_locked(&nd.anim, ch);
                        crate::anim::set_locked(&mut nd.anim, ch, !locked);
                    }
                }
            }
        }

        // --- clicks ---
        if !modal && resp.clicked() {
            if let Some(p) = pointer {
                if ruler.contains(p) {
                    seek = Some(self.frame_at(&area, p.x).round().max(0.0) as u64);
                } else if list.contains(p) {
                    // The list edits the SHARED view, with the curve list's
                    // gestures: click isolates (again = show all), shift-click
                    // hides. A node row's arrow folds it here only.
                    let kind = row_at(p.y, self).map(|i| rows[i].kind);
                    let these: Vec<(u64, Channel)> = match kind {
                        Some(RowKind::Node(id)) => {
                            listed.iter().copied().filter(|k| k.0 == id && !view.hidden.contains(k)).collect()
                        }
                        Some(RowKind::Chan(n, ch)) => vec![(n, ch)],
                        _ => Vec::new(),
                    };
                    if let (Some(RowKind::Node(id)), true) = (kind, p.x < list.min.x + 20.0) {
                        if !self.collapsed.remove(&id) {
                            self.collapsed.insert(id);
                        }
                    } else if !these.is_empty() && mods.shift {
                        view.hidden.extend(these);
                    } else if !these.is_empty() {
                        let visible = view.visible(&listed);
                        if visible.len() == these.len() {
                            view.hidden.clear();
                        } else {
                            view.hidden = listed.iter().copied().filter(|k| !these.contains(k)).collect();
                        }
                    }
                } else if let Some(hit) = key_at(p, self) {
                    let under = pick(self, graph, hit);
                    if mods.shift {
                        let all_in = under.iter().all(|k| self.selection.contains(k));
                        for k in under {
                            if all_in {
                                self.selection.remove(&k);
                            } else {
                                self.selection.insert(k);
                            }
                        }
                    } else {
                        self.selection = under.into_iter().collect();
                    }
                } else if !mods.shift {
                    self.selection.clear();
                }
            }
        }

        // --- paint: ruler + grid ---
        let fstep = nice_step(70.0 / self.x_scale).max(1.0);
        let grid = Stroke::new(1.0, theme::grid_line());
        let mut f = (self.frame_at(&area, area.min.x) / fstep).floor() * fstep;
        while self.x_of(&area, f) < area.max.x {
            let x = self.x_of(&area, f);
            if x >= area.min.x {
                painter.line_segment([Pos2::new(x, ruler.max.y), Pos2::new(x, area.max.y)], grid);
                painter.text(
                    Pos2::new(x + 3.0, ruler.center().y),
                    egui::Align2::LEFT_CENTER,
                    format!("{}", f as i64),
                    egui::FontId::proportional(10.0),
                    theme::text_weak(),
                );
            }
            f += fstep;
        }

        // --- paint: rows ---
        let clip_area = painter.with_clip_rect(area);
        let clip_list = painter.with_clip_rect(Rect::from_min_max(Pos2::new(list.min.x, area.min.y), list.max));
        for (i, row) in rows.iter().enumerate() {
            let y0 = row_top(i, self);
            if y0 + ROW_H < area.min.y || y0 > area.max.y {
                continue;
            }
            let band = Rect::from_min_max(Pos2::new(full.min.x, y0), Pos2::new(full.max.x, y0 + ROW_H));
            let (fill, indent, strong) = match row.kind {
                RowKind::Summary => (Some(theme::raised_fill()), 6.0, true),
                RowKind::Node(_) => (Some(theme::hover_fill()), 6.0, true),
                RowKind::Chan(..) => ((i % 2 == 0).then(theme::hover_fill).map(|c| c.gamma_multiply(0.4)), 22.0, false),
            };
            if let Some(c) = fill {
                clip_area.rect_filled(band, 0.0, c);
                clip_list.rect_filled(band, 0.0, c);
            }
            let mut indent = indent;
            if let RowKind::Node(id) = row.kind {
                // Fold arrow, painted: the UI font has no triangle glyphs.
                let c = Pos2::new(list.min.x + 11.0, y0 + ROW_H * 0.5);
                let pts = if self.collapsed.contains(&id) {
                    vec![c + egui::vec2(-2.5, -4.0), c + egui::vec2(3.5, 0.0), c + egui::vec2(-2.5, 4.0)]
                } else {
                    vec![c + egui::vec2(-4.0, -2.5), c + egui::vec2(4.0, -2.5), c + egui::vec2(0.0, 3.5)]
                };
                clip_list.add(egui::Shape::convex_polygon(pts, theme::text(), Stroke::NONE));
                indent += 14.0;
            }
            let text = match row.kind {
                RowKind::Chan(..) if row.locked => format!("{} ·lock", row.label),
                _ => row.label.clone(),
            };
            let col = if row.locked {
                theme::locked()
            } else if strong {
                theme::text_strong()
            } else {
                theme::text()
            };
            clip_list.text(
                Pos2::new(list.min.x + indent, y0 + ROW_H * 0.5),
                egui::Align2::LEFT_CENTER,
                text,
                egui::FontId::proportional(11.0),
                col,
            );

            // Hold bars: two neighbouring keys with the same value on a
            // channel row, like Blender's — "nothing changes here".
            let yc = y0 + ROW_H * 0.5;
            if let RowKind::Chan(n, ch) = row.kind {
                if let Some(c) = graph.nodes.get(&n).and_then(|nd| nd.anim.get(&ch)) {
                    for w in c.keys.windows(2) {
                        if (w[0].value - w[1].value).abs() <= 1e-6 * w[0].value.abs().max(1.0) {
                            let (xa, xb) = (self.x_of(&area, w[0].frame as f32), self.x_of(&area, w[1].frame as f32));
                            clip_area.rect_filled(
                                Rect::from_min_max(Pos2::new(xa, yc - 2.5), Pos2::new(xb, yc + 2.5)),
                                1.0,
                                theme::text_weak().gamma_multiply(0.5),
                            );
                        }
                    }
                }
            }
            // Diamonds. A group diamond is selected when every key under it is.
            let size = match row.kind {
                RowKind::Chan(..) => 5.0,
                _ => 5.5,
            };
            for f in &row.frames {
                let x = self.x_of(&area, *f as f32);
                if x < area.min.x - 8.0 || x > area.max.x + 8.0 {
                    continue;
                }
                let under = self.keys_under(graph, &all_chans, row.kind, *f);
                let selected = !under.is_empty() && under.iter().all(|k| self.selection.contains(k));
                let base = if row.locked {
                    theme::locked()
                } else if matches!(row.kind, RowKind::Chan(..)) {
                    theme::text_strong()
                } else {
                    theme::text()
                };
                let fillc = if selected { theme::keyframe() } else { base };
                let p = Pos2::new(x, yc);
                let hot = pointer.is_some_and(|q| row_at(q.y, self) == Some(i) && (q.x - x).abs() <= HIT && area.contains(q));
                let d = if hot { size + 1.0 } else { size };
                clip_area.add(egui::Shape::convex_polygon(
                    vec![Pos2::new(p.x, p.y - d), Pos2::new(p.x + d, p.y), Pos2::new(p.x, p.y + d), Pos2::new(p.x - d, p.y)],
                    fillc,
                    Stroke::new(1.0, Color32::from_black_alpha(160)),
                ));
            }
        }
        painter.line_segment([Pos2::new(list.max.x, full.min.y), Pos2::new(list.max.x, full.max.y)], Stroke::new(1.0, theme::border()));
        painter.line_segment([Pos2::new(full.min.x, ruler.max.y), Pos2::new(full.max.x, ruler.max.y)], Stroke::new(1.0, theme::border()));

        // --- playhead (+ a pivot hint while scaling) ---
        let px = self.x_of(&area, t as f32);
        if px >= area.min.x && px <= area.max.x {
            let ph = theme::timeline_playhead();
            painter.line_segment([Pos2::new(px, ruler.min.y), Pos2::new(px, area.max.y)], Stroke::new(1.5, ph));
            let tag = Rect::from_center_size(Pos2::new(px, ruler.center().y), egui::vec2(30.0, RULER_H - 4.0));
            painter.rect_filled(tag, 3.0, ph);
            painter.text(tag.center(), egui::Align2::CENTER_CENTER, format!("{t}"), egui::FontId::proportional(10.0), Color32::WHITE);
        }
        if let Some(m) = marquee {
            painter.rect_filled(m, 0.0, theme::marquee_fill());
            painter.rect_stroke(m, 0.0, Stroke::new(1.0, theme::marquee()), egui::StrokeKind::Outside);
        }

        // --- status ---
        let mut status = if self.op.as_ref().is_some_and(|o| matches!(o, Op::Move { modal: true, .. } | Op::Scale { modal: true, .. })) {
            format!("{} · click/Enter confirm · RMB/Esc cancel", self.note)
        } else if !self.selection.is_empty() {
            format!("{} key(s) selected", self.selection.len())
        } else {
            String::new()
        };
        if let Some(p) = pointer.filter(|p| area.contains(*p)) {
            status = format!("{status}   f {}", self.frame_at(&area, p.x).round() as i64);
        }
        if !status.is_empty() {
            let at = Pos2::new(full.max.x - 6.0, full.max.y - 4.0);
            painter.text(
                at,
                egui::Align2::RIGHT_BOTTOM,
                status.trim(),
                egui::FontId::proportional(10.0),
                theme::text(),
            );
        }
        if self.op.is_some() || marquee.is_some() {
            ui.ctx().request_repaint();
        }
        seek
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    fn graph(frames: &[u64]) -> (EditorGraph, u64) {
        let mut g = EditorGraph::default();
        let id = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let c = g.nodes.get_mut(&id).unwrap().anim.entry(Channel::TranslateX).or_default();
        for (i, f) in frames.iter().enumerate() {
            c.set_key(*f, i as f32);
        }
        (g, id)
    }
    fn frames(g: &EditorGraph, id: u64) -> Vec<u64> {
        g.nodes[&id].anim.get(&Channel::TranslateX).map_or(vec![], |c| c.keys.iter().map(|k| k.frame as u64).collect())
    }
    fn sel(id: u64, fs: &[u64]) -> HashSet<KeyId> {
        fs.iter().map(|f| (id, Channel::TranslateX, frame_id(*f as f64))).collect()
    }

    #[test]
    fn move_is_exact_from_the_snapshot_and_keeps_interp() {
        let (mut g, id) = graph(&[0, 10, 20]);
        g.nodes.get_mut(&id).unwrap().anim.get_mut(&Channel::TranslateX).unwrap().set_interp(10, Interp::Hold);
        let (snap, grabbed) = grab(&g, &sel(id, &[10]));
        move_keys(&mut g, &snap, &grabbed, 3, false);
        let s = move_keys(&mut g, &snap, &grabbed, 5, false);
        assert_eq!(frames(&g, id), vec![0, 15, 20]);
        assert_eq!(s, sel(id, &[15]));
        let k = g.nodes[&id].anim[&Channel::TranslateX].keys[1];
        assert_eq!((k.value, k.interp), (1.0, Interp::Hold));
    }

    #[test]
    fn landing_on_a_key_replaces_it_and_cancel_restores() {
        let (mut g, id) = graph(&[0, 10, 20]);
        let (snap, grabbed) = grab(&g, &sel(id, &[10]));
        move_keys(&mut g, &snap, &grabbed, 10, false);
        assert_eq!(frames(&g, id), vec![0, 20]);
        assert_eq!(g.nodes[&id].anim[&Channel::TranslateX].keys[1].value, 1.0);
        restore(&mut g, &snap);
        assert_eq!(frames(&g, id), vec![0, 10, 20]);
    }

    #[test]
    fn negative_moves_clamp_at_zero() {
        let (mut g, id) = graph(&[5]);
        let (snap, grabbed) = grab(&g, &sel(id, &[5]));
        move_keys(&mut g, &snap, &grabbed, -50, false);
        assert_eq!(frames(&g, id), vec![0]);
    }

    #[test]
    fn duplicate_keeps_the_originals() {
        let (mut g, id) = graph(&[0, 10]);
        let (snap, grabbed) = grab(&g, &sel(id, &[0, 10]));
        let s = move_keys(&mut g, &snap, &grabbed, 30, true);
        assert_eq!(frames(&g, id), vec![0, 10, 30, 40]);
        assert_eq!(s, sel(id, &[30, 40]));
    }

    #[test]
    fn scale_about_the_playhead() {
        let (mut g, id) = graph(&[10, 20, 30]);
        let (snap, grabbed) = grab(&g, &sel(id, &[10, 20, 30]));
        scale_keys(&mut g, &snap, &grabbed, 10.0, 2.0);
        assert_eq!(frames(&g, id), vec![10, 30, 50]);
    }

    #[test]
    fn delete_drops_empty_curves_but_keeps_locks() {
        let (mut g, id) = graph(&[0, 10]);
        delete_keys(&mut g, sel(id, &[0, 10]));
        assert!(g.nodes[&id].anim.get(&Channel::TranslateX).is_none());
        let (mut g, id) = graph(&[0]);
        g.nodes.get_mut(&id).unwrap().anim.get_mut(&Channel::TranslateX).unwrap().locked = true;
        delete_keys(&mut g, sel(id, &[0]));
        assert!(g.nodes[&id].anim[&Channel::TranslateX].locked);
    }

    #[test]
    fn locked_channels_are_never_grabbed_or_pasted() {
        let (mut g, id) = graph(&[0, 10]);
        g.nodes.get_mut(&id).unwrap().anim.get_mut(&Channel::TranslateX).unwrap().locked = true;
        let (_, grabbed) = grab(&g, &sel(id, &[0, 10]));
        assert!(grabbed.is_empty());
        let clip = vec![(id, Channel::TranslateX, Key::new(0, 9.0))];
        assert!(paste_keys(&mut g, &clip, 50).is_empty());
        assert_eq!(frames(&g, id), vec![0, 10]);
    }

    #[test]
    fn paste_lands_relative_to_the_playhead() {
        let (mut g, id) = graph(&[0]);
        let clip = vec![(id, Channel::TranslateX, Key::new(0, 1.0)), (id, Channel::TranslateX, Key::new(4, 2.0))];
        let s = paste_keys(&mut g, &clip, 100);
        assert_eq!(frames(&g, id), vec![0, 100, 104]);
        assert_eq!(s, sel(id, &[100, 104]));
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use engine::graph::NodeKind;

    /// Look at it: `cargo test -p app dopesheet_render -- --ignored`,
    /// then open `target/dopesheet.png`.
    #[test]
    #[ignore]
    fn dopesheet_render() {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let b = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        for (id, ch, keys) in [
            (a, Channel::TranslateX, vec![(0, 0.0), (12, 40.0), (24, 40.0), (40, 0.0)]),
            (a, Channel::Rotate, vec![(6, 0.0), (30, 90.0)]),
            (b, Channel::ScaleX, vec![(10, 1.0), (20, 2.0), (30, 1.0)]),
        ] {
            let c = g.nodes.get_mut(&id).unwrap().anim.entry(ch).or_default();
            for (f, v) in keys {
                c.set_key(f, v);
            }
        }
        let mut ds = DopeSheet::default();
        // Hidden in the curve editor = not a row here; saved sets show in the header.
        let mut view = ChannelView::default();
        view.hidden.insert((a, Channel::Rotate));
        view.sets.push(crate::chanview::ChanSet { name: "moves".into(), chans: vec![(a, Channel::TranslateX)] });
        ds.selection.insert((a, Channel::TranslateX, frame_id(12.0)));
        ds.selection.insert((a, Channel::TranslateX, frame_id(24.0)));
        let out = crate::uishot::render("dopesheet", 900.0, 220.0, 2.0, 3, |ui, _| {
            ds.ui(ui, &mut g, &mut view, &[], 18);
        });
        if let Some(p) = out {
            eprintln!("wrote {}", p.display());
        }
    }
}
