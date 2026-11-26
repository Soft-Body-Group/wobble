//! Mouse editing on the timeline, beside the keys (which all still work).
//! What a press does depends on the TOOL (V C R Y U, or the strip at the
//! timeline's top left).
//!
//! Select (V):
//! - **Click** a clip to select it (shift / cmd: add or remove), click an
//!   edge to select that edit point, click an empty lane or the ruler to
//!   move the playhead there (an empty lane also clears the selection).
//!   Drag from an empty lane: a MARQUEE selects every clip it touches
//!   (shift: add to the selection).
//! - **Drag a clip** to move it in time and across tracks; its linked sound
//!   moves with it to the paired track (V2 -> A2). alt at the press:
//!   DUPLICATE (the copy moves, the original stays); alt+shift: the clip
//!   alone, without its sound. Dropping overwrites what is under it; cmd
//!   held inserts (every track opens a gap). The record viewer shows the
//!   frame at the dragged clip's head.
//! - **Drag an edge** to trim that side (the cut moves; a gap opens or
//!   closes), ripple when the trim mode is ripple (B) or cmd is held. The
//!   two viewers become the trim monitor while it runs, and the source the
//!   clip has beyond its edges shows as a blue band with its frame count.
//! - **Drag a dissolve's edge** to change that side's length.
//!
//! HOLD keys (quasimodes; they beat the tool, and the edge grabbed does not
//! matter): while one is held, press anywhere on a clip and drag —
//!   A  trim its in point (plain)        D  trim its out point (plain)
//!   Z  ripple its in point              C  ripple its out point
//!   S  slip it
//!   shift A  rate-stretch from its in   shift D  rate-stretch from its out
//!   shift S  its speed, scaled about the playhead (live speed shown)
//! The label by the pointer names the op while the key is down. A held key
//! that dragged fires nothing when let go; D tapped alone still selects.
//!
//! Roll (N): drag a cut to move it, one clip longer and the other shorter.
//! Razor (R over the timeline): click cuts the clip under the pointer there (and its sound);
//! shift-click cuts every track. Rate stretch (R): drag an edge and the
//! clip's speed changes so the same source fills the new length. Slip (Y):
//! drag a clip to show other source in the same place. Slide (U): drag a
//! clip along; the clips either side give and take.
//!
//! Dropped on the timeline: the source viewer's clip or a bin file (dragged
//! from the picture / the bin) overwrites there on the row it is dropped on
//! (cmd: insert); the cross dissolve (from the bin's transitions) goes on
//! the cut nearest the drop.
//! - **Snapping** (T toggles): the dragged edge(s) jump to the playhead, the
//!   in/out marks and every cut within a few pixels; a line shows where.
//!
//! Every drag is computed from a snapshot of the sequence taken at its press
//! with the model's own edits (`move_clips`, `trim`, `set_dissolve_sides`),
//! so it previews live, Esc puts the snapshot back, and the release is ONE
//! undo step. Nothing is decoded that playback would not: the record viewer
//! asks for one frame, the dragged clip's head.

use crate::editor::{Editor, Focus};
use crate::timelineui::{Cmd, Event, Hit, Overlay, Payload, Selection, Tool};
use model::{Sequence, Side};

/// What a held key makes a drag on a clip do (see the module comment).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldOp {
    TrimIn,
    TrimOut,
    RippleIn,
    RippleOut,
    Slip,
    StretchIn,
    StretchOut,
    Speed,
}

impl HoldOp {
    /// The op for a held key (with or without shift); none for shift+Z/C.
    pub fn of(k: egui::Key, shift: bool) -> Option<HoldOp> {
        use egui::Key::*;
        Some(match (k, shift) {
            (A, false) => HoldOp::TrimIn,
            (D, false) => HoldOp::TrimOut,
            (Z, false) => HoldOp::RippleIn,
            (C, false) => HoldOp::RippleOut,
            (S, false) => HoldOp::Slip,
            (A, true) => HoldOp::StretchIn,
            (D, true) => HoldOp::StretchOut,
            (S, true) => HoldOp::Speed,
            _ => return None,
        })
    }

    /// The pointer's label.
    pub fn label(self) -> &'static str {
        match self {
            HoldOp::TrimIn => "trim in (A)",
            HoldOp::TrimOut => "trim out (D)",
            HoldOp::RippleIn => "ripple in (Z)",
            HoldOp::RippleOut => "ripple out (C)",
            HoldOp::Slip => "slip (S)",
            HoldOp::StretchIn => "stretch in (shift A)",
            HoldOp::StretchOut => "stretch out (shift D)",
            HoldOp::Speed => "speed at playhead (shift S)",
        }
    }
}

/// Pixels a dragged edge snaps across.
const SNAP_PX: f64 = 8.0;

/// A drag in progress.
pub struct Drag {
    /// The sequence as it was at the press.
    snapshot: Sequence,
    kind: Kind,
    /// The selection at the press (put back on cancel).
    sel: Selection,
}

enum Kind {
    /// Clips (track, index in the snapshot), the one under the press first;
    /// `dup`: copies move, the originals stay.
    Move { picks: Vec<(usize, usize)>, head: (usize, usize), dup: bool },
    /// Rate stretch of a clip's edge.
    Stretch { track: usize, idx: usize, side: Side },
    Slip { track: usize, idx: usize },
    Slide { track: usize, idx: usize },
    /// The speed, scaled about a record frame (shift+S held).
    Speed { track: usize, idx: usize, anchor: i64 },
    Trim { track: usize, at: i64, side: Side, ripple: bool },
    Dissolve { track: usize, at: i64, end: bool },
}

impl Editor {
    /// Handle what the timeline reported this frame. True when it was a
    /// mouse edit (the caller then skips its scrub handling).
    pub(crate) fn timeline_mouse(&mut self, ev: &Event) -> bool {
        match ev {
            Event::Click { hit, add, frame, mods } => {
                if self.rec.tool == Tool::Razor {
                    self.razor_click(*hit, *frame, mods.shift);
                } else {
                    self.click(*hit, *add, *frame);
                }
                true
            }
            Event::Drag { hit, df, dtrack, mods, done } => {
                if self.drag_cancelled {
                    if *done {
                        self.drag_cancelled = false;
                    }
                    return true;
                }
                if self.drag.is_none() {
                    self.drag_start(*hit, *mods);
                }
                self.drag_update(*df, *dtrack, mods.command);
                if *done {
                    self.drag_end();
                }
                true
            }
            Event::Marquee { picks, add, done } => {
                self.focus = Focus::Record;
                self.monitor = Focus::Record;
                let mut v: Vec<(usize, i64)> = match (&self.rec.sel, *add) {
                    (Selection::Clips(v), true) => v.clone(),
                    (Selection::Clip { track, start }, true) => vec![(*track, *start)],
                    _ => Vec::new(),
                };
                for p in picks {
                    if !v.contains(p) {
                        v.push(*p);
                    }
                }
                self.rec.sel = match v.len() {
                    0 => Selection::None,
                    1 => Selection::Clip { track: v[0].0, start: v[0].1 },
                    _ => Selection::Clips(v),
                };
                if *done {
                    let n = match &self.rec.sel {
                        Selection::Clips(v) => v.len(),
                        Selection::Clip { .. } => 1,
                        _ => 0,
                    };
                    self.say(format!("{n} clip{} selected", if n == 1 { "" } else { "s" }));
                }
                true
            }
            Event::Drop { payload, frame, track, insert } => {
                self.drop_on_timeline(payload.clone(), *frame, *track, *insert);
                true
            }
            Event::Menu(c) => {
                self.menu_cmd(*c);
                true
            }
            Event::Open { track, start } => {
                if self.rec.seq.tracks[*track].clips.iter().any(|c| c.start == *start && c.text.is_some()) {
                    self.rec.sel = Selection::Clip { track: *track, start: *start };
                    self.open_text(*track, *start);
                }
                true
            }
            Event::Tool(t) => {
                self.set_tool(*t);
                true
            }
            _ => false,
        }
    }

    pub(crate) fn set_tool(&mut self, t: Tool) {
        self.rec.tool = t;
        let k = if t.key().is_empty() { "hold shift A / D drags do it too" } else { t.key() };
        self.say(format!("{} tool ({k})", t.name()));
    }

    /// The razor's frame: the pointer's, pulled onto the playhead or a cut
    /// within the snap distance (snapping on).
    fn razor_frame(&self, frame: f64) -> i64 {
        let f = frame.round() as i64;
        let (s, _) = self.snap(&[f], &[], &self.rec.seq.clone());
        f + s
    }

    fn razor_click(&mut self, hit: Hit, frame: f64, all: bool) {
        self.focus = Focus::Record;
        self.monitor = Focus::Record;
        let at = self.razor_frame(frame);
        let track = match hit {
            Hit::Clip { track, .. } | Hit::Edge { track, .. } | Hit::Dissolve { track, .. } => Some(track),
            Hit::Empty => None,
        };
        self.razor_at(track, at, all);
    }

    /// Cut at `at`: on `track` (with its linked group), or every track.
    pub(crate) fn razor_at(&mut self, track: Option<usize>, at: i64, all: bool) {
        let n = if all {
            let every: Vec<usize> = (0..self.rec.seq.tracks.len()).collect();
            self.rec.seq.razor_tracks(&every, at)
        } else if let Some(t) = track {
            self.rec.seq.razor(t, at) as usize
        } else {
            0
        };
        if n == 0 {
            self.say("nothing to cut there");
            return;
        }
        self.rec.sel = Selection::None;
        self.changed(false);
        let tc = crate::editor::timecode(at.max(0) as u64, self.rec.seq.fps.as_f64());
        self.say(if all { format!("cut every track at {tc}") } else { format!("cut at {tc}") });
    }

    fn click(&mut self, hit: Hit, add: bool, frame: f64) {
        self.focus = Focus::Record;
        self.monitor = Focus::Record;
        match hit {
            Hit::Clip { track, start } => {
                self.rec.sel = match (&self.rec.sel, add) {
                    (Selection::Clip { track: t, start: s }, true) if (*t, *s) != (track, start) => Selection::Clips(vec![(*t, *s), (track, start)]),
                    (Selection::Clips(v), true) => {
                        let mut v = v.clone();
                        match v.iter().position(|&x| x == (track, start)) {
                            Some(i) => {
                                v.remove(i);
                            }
                            None => v.push((track, start)),
                        }
                        Selection::Clips(v)
                    }
                    _ => Selection::Clip { track, start },
                };
            }
            Hit::Edge { track, at, side } => {
                self.rec.sel = Selection::Edit { track, at, side };
                self.rec.transport.stop();
                self.rec.transport.seek(at.max(0) as u64);
            }
            Hit::Dissolve { track, at, .. } => {
                self.rec.sel = Selection::Edit { track, at, side: Side::Roll };
            }
            Hit::Empty => {
                // An empty lane: deselect, and the playhead goes there (past
                // the last clip too).
                self.rec.sel = Selection::None;
                self.rec.transport.stop();
                let f = frame.floor().max(0.0) as u64;
                self.rec.transport.seek(f);
            }
        }
    }

    /// The right-click menu's commands.
    pub(crate) fn menu_cmd(&mut self, c: Cmd) {
        self.focus = Focus::Record;
        self.monitor = Focus::Record;
        let seq = &self.rec.seq;
        let idx = |t: usize, st: i64| seq.tracks[t].clips.iter().position(|c| c.start == st);
        match c {
            Cmd::AddDissolve { track, at } => self.dissolve_at_cut(track, at, None),
            Cmd::Input { media, space } => {
                let p = self.rec.seq.media[media].path.clone();
                self.set_input_space(&p, space);
            }
            Cmd::RemoveDissolve { track, at } => {
                if self.rec.seq.remove_dissolve(track, at) {
                    self.changed(false);
                    self.say("dissolve removed");
                }
            }
            Cmd::SelectEdit { track, at } => {
                self.rec.sel = Selection::Edit { track, at, side: Side::Roll };
                self.rec.transport.stop();
                self.rec.transport.seek(at.max(0) as u64);
            }
            Cmd::Razor { track, at, all } => self.razor_at(Some(track), at, all),
            Cmd::Lift { track, start } | Cmd::RippleDelete { track, start } | Cmd::Speed { track, start } => {
                if idx(track, start).is_none() {
                    return;
                }
                self.rec.sel = Selection::Clip { track, start };
                match c {
                    Cmd::Lift { .. } => self.delete_clip(false),
                    Cmd::RippleDelete { .. } => self.delete_clip(true),
                    _ => self.ask_speed(),
                }
            }
        }
    }

    /// A payload dropped on the timeline at `frame` on the row of `track`.
    pub(crate) fn drop_on_timeline(&mut self, payload: Payload, frame: i64, track: Option<usize>, insert: bool) {
        self.focus = Focus::Record;
        self.monitor = Focus::Record;
        match &payload {
            Payload::Dissolve => {
                let Some(t) = track else { return };
                match self.rec.seq.nearest_cut(t, frame) {
                    Some(at) => self.dissolve_at_cut(t, at, None),
                    None => self.say("drop the dissolve on a cut"),
                }
            }
            Payload::Source { .. } => self.drop_edit(frame, track, insert),
            Payload::Text => match track.filter(|t| model::VIDEO_TRACKS.contains(t)) {
                Some(t) => {
                    self.new_text(Some(t), Some(frame), "Text");
                }
                None => self.say("text goes on a video track (V1-V3)"),
            },
            Payload::File(p) => {
                self.drop_file(p, frame, track, insert);
            }
            Payload::Files(ps) => {
                // One after another from the drop, each the whole clip.
                let mut at = frame;
                for p in ps {
                    match self.drop_file(p, at, track, insert) {
                        Some(len) => at += len,
                        None => break,
                    }
                }
                self.say(format!("{} clips laid down from {}", ps.len(), crate::editor::timecode(frame as u64, self.rec.seq.fps.as_f64())));
            }
            Payload::Clips(cs) => {
                let mut at = frame;
                for (p, r) in cs {
                    match self.drop_clip(p, *r, at, track, insert) {
                        Some(len) => at += len,
                        None => break,
                    }
                }
                let what = if cs.len() == 1 { "subclip".to_string() } else { format!("{} clips", cs.len()) };
                self.say(format!("{what} laid down from {}", crate::editor::timecode(frame as u64, self.rec.seq.fps.as_f64())));
            }
        }
    }

    /// A whole file dropped at `frame`: loaded into the source (no marks)
    /// and edited in. The record frames it took, or `None` if it would not
    /// load.
    fn drop_file(&mut self, p: &std::path::Path, frame: i64, track: Option<usize>, insert: bool) -> Option<i64> {
        self.drop_clip(p, None, frame, track, insert)
    }

    /// A file dropped at `frame`, the whole clip or (`range`) a subclip's
    /// in..out: loaded into the source with those marks and edited in.
    fn drop_clip(&mut self, p: &std::path::Path, range: Option<(i64, i64)>, frame: i64, track: Option<usize>, insert: bool) -> Option<i64> {
        self.open(p);
        if !self.source.as_ref().is_some_and(|c| c.original == p) {
            return None;
        }
        if let Some(c) = &mut self.source {
            c.marks = range.map_or(model::Marks::default(), |(a, b)| model::Marks { inp: Some(a), out: Some(b) });
            c.transport.seek(range.map_or(0, |(a, _)| a.max(0) as u64));
        }
        let t = track?;
        self.drop_edit(frame, Some(t), insert);
        // A sound file dropped on a video row lands on its paired audio track.
        let on = if model::VIDEO_TRACKS.contains(&t) && self.source.as_ref().is_some_and(|c| c.info.audio_only) { model::audio_of(t) } else { t };
        self.rec.seq.tracks[on].clips.iter().find(|c| c.start == frame).map(|c| c.len)
    }

    /// Insert / overwrite the source at `frame` on the dropped-on row:
    /// a video row takes the picture (and the sound to its paired track),
    /// an audio row the sound alone. Record marks and targets are left as
    /// they were, and so is the playhead.
    fn drop_edit(&mut self, frame: i64, track: Option<usize>, insert: bool) {
        let Some(t) = track else { return };
        let keep = (self.rec.marks, self.rec.targets, self.rec.transport.frame());
        self.rec.targets = if model::VIDEO_TRACKS.contains(&t) {
            model::Targets { video: Some(t), audio: Some(model::audio_of(t)) }
        } else {
            model::Targets { video: None, audio: Some(t) }
        };
        self.rec.marks = model::Marks { inp: Some(frame), out: None };
        self.edit(insert);
        self.rec.marks = keep.0;
        self.rec.targets = keep.1;
        self.rec.transport.seek(keep.2);
        let tc = crate::editor::timecode(frame as u64, self.rec.seq.fps.as_f64());
        self.say(format!("{} at {tc} on {}", if insert { "inserted" } else { "overwrote" }, self.rec.seq.tracks[t].name));
    }

    /// The clips a move takes: the selection when the press was on a
    /// selected clip, else the clip pressed; each with its linked group
    /// unless alt was held.
    fn move_picks(&self, track: usize, start: i64, alone: bool) -> Option<(Vec<(usize, usize)>, (usize, usize))> {
        let seq = &self.rec.seq;
        let find = |t: usize, s: i64| seq.tracks[t].clips.iter().position(|c| c.start == s).map(|i| (t, i));
        let head = find(track, start)?;
        let chosen: Vec<(usize, i64)> = match &self.rec.sel {
            Selection::Clips(v) if v.contains(&(track, start)) => v.clone(),
            _ => vec![(track, start)],
        };
        let mut picks = Vec::new();
        for (t, s) in chosen {
            let Some(p) = find(t, s) else { continue };
            let group = if alone { vec![p] } else { seq.group(p.0, p.1) };
            for g in group {
                if !picks.contains(&g) {
                    picks.push(g);
                }
            }
        }
        Some((picks, head))
    }

    fn drag_start(&mut self, hit: Hit, mods: egui::Modifiers) {
        let seq = &self.rec.seq;
        let find = |t: usize, s: i64| seq.tracks[t].clips.iter().position(|c| c.start == s);
        // The clip a slip / slide / stretch acts on: the one pressed, or for
        // an edge the clip on that side of it.
        let clip_of = |h: Hit| match h {
            Hit::Clip { track, start } => find(track, start).map(|i| (track, i, Side::Roll)),
            Hit::Edge { track, at, side: Side::Out } => seq.tracks[track].cut(at).0.map(|i| (track, i, Side::Out)),
            Hit::Edge { track, at, side } => seq.tracks[track].cut(at).1.map(|i| (track, i, side)),
            _ => None,
        };
        // A held key beats the tool, wherever on the clip the press is.
        let held = self.hold.and_then(|op| {
            let (track, idx) = match hit {
                Hit::Clip { track, start } => (track, find(track, start)?),
                Hit::Edge { track, at, side: Side::Out } => (track, seq.tracks[track].cut(at).0?),
                Hit::Edge { track, at, .. } => (track, seq.tracks[track].cut(at).1?),
                _ => return None,
            };
            let c = &seq.tracks[track].clips[idx];
            Some(match op {
                HoldOp::TrimIn => Kind::Trim { track, at: c.start, side: Side::In, ripple: false },
                HoldOp::TrimOut => Kind::Trim { track, at: c.end(), side: Side::Out, ripple: false },
                HoldOp::RippleIn => Kind::Trim { track, at: c.start, side: Side::In, ripple: true },
                HoldOp::RippleOut => Kind::Trim { track, at: c.end(), side: Side::Out, ripple: true },
                HoldOp::Slip => Kind::Slip { track, idx },
                HoldOp::StretchIn => Kind::Stretch { track, idx, side: Side::In },
                HoldOp::StretchOut => Kind::Stretch { track, idx, side: Side::Out },
                HoldOp::Speed => Kind::Speed { track, idx, anchor: self.rec.transport.frame() as i64 },
            })
        });
        if held.is_some() {
            self.hold_used = true;
        }
        let kind = match (self.rec.tool, hit) {
            _ if held.is_some() => held.expect("checked"),
            (Tool::Razor, _) => return,
            (Tool::Slip, h) => match clip_of(h) {
                Some((track, idx, _)) => Kind::Slip { track, idx },
                None => return,
            },
            (Tool::Slide, h) => match clip_of(h) {
                Some((track, idx, _)) => Kind::Slide { track, idx },
                None => return,
            },
            (Tool::Rate, h @ Hit::Edge { .. }) => match clip_of(h) {
                Some((track, idx, side)) => Kind::Stretch { track, idx, side },
                None => return,
            },
            (Tool::Rate, _) => return,
            (Tool::Select, Hit::Clip { track, start }) => {
                let Some((picks, head)) = self.move_picks(track, start, mods.alt && mods.shift) else { return };
                if !matches!(&self.rec.sel, Selection::Clips(v) if v.contains(&(track, start))) {
                    self.rec.sel = Selection::Clip { track, start };
                }
                Kind::Move { picks, head, dup: mods.alt && !mods.shift }
            }
            // A plain edge drag always trims that one side (the cut moves,
            // a gap opens or closes); ripple only in ripple mode (B) or with
            // cmd. Rolling is the roll tool's (N), never a hidden mode.
            (Tool::Select, Hit::Edge { track, at, side }) => Kind::Trim { track, at, side, ripple: self.rec.ripple || mods.command },
            (Tool::Roll, Hit::Edge { track, at, .. }) | (Tool::Roll, Hit::Dissolve { track, at, .. }) => Kind::Trim { track, at, side: Side::Roll, ripple: false },
            (Tool::Roll, _) => return,
            (Tool::Select, Hit::Dissolve { track, at, end }) => Kind::Dissolve { track, at, end },
            (Tool::Select, Hit::Empty) => return,
        };
        // A slip / slide / stretch selects its clip.
        if let Kind::Slip { track, idx } | Kind::Slide { track, idx } | Kind::Stretch { track, idx, .. } | Kind::Speed { track, idx, .. } = kind {
            self.rec.sel = Selection::Clip { track, start: self.rec.seq.tracks[track].clips[idx].start };
        }
        self.focus = Focus::Record;
        self.monitor = Focus::Record;
        self.rec.transport.stop();
        self.drag = Some(Drag { snapshot: self.rec.seq.clone(), kind, sel: self.rec.sel.clone() });
    }

    /// Snap `edges` (frames, all moved by the same amount) to the nearest
    /// target within `SNAP_PX`: the shift to add, and where the line goes.
    fn snap(&self, edges: &[i64], skip: &[i64], snapshot: &Sequence) -> (i64, Option<i64>) {
        if !self.rec.snap {
            return (0, None);
        }
        let mut targets = snapshot.edit_points();
        targets.push(self.rec.transport.frame() as i64);
        targets.extend(self.rec.marks.inp);
        targets.extend(self.rec.marks.out.map(|o| o + 1));
        targets.retain(|t| !skip.contains(t));
        let reach = (SNAP_PX * self.rec.view.fpp()).max(0.5);
        let mut best: Option<(i64, i64)> = None;
        for &e in edges {
            for &t in &targets {
                let d = t - e;
                if (d as f64).abs() <= reach && best.is_none_or(|(b, _)| d.abs() < b.abs()) {
                    best = Some((d, t));
                }
            }
        }
        best.map_or((0, None), |(d, t)| (d, Some(t)))
    }

    fn drag_update(&mut self, df: f64, dtrack: i32, insert: bool) {
        let Some(drag) = self.drag.take() else { return };
        let mut seq = drag.snapshot.clone();
        let d = df.round() as i64;
        let mut overlay = Overlay::default();
        match &drag.kind {
            Kind::Move { picks, head, dup } => {
                let clips: Vec<_> = picks.iter().map(|&(t, i)| drag.snapshot.tracks[t].clips[i].clone()).collect();
                let starts: Vec<i64> = clips.iter().map(|c| c.start).collect();
                let ends: Vec<i64> = clips.iter().map(|c| c.end()).collect();
                let edges: Vec<i64> = starts.iter().chain(&ends).map(|e| e + d).collect();
                let own: Vec<i64> = starts.iter().chain(&ends).copied().collect();
                let (s, line) = self.snap(&edges, &own, &drag.snapshot);
                let (dt, dtr) = if *dup { seq.duplicate_clips(picks, d + s, dtrack, insert) } else { seq.move_clips(picks, d + s, dtrack, insert) };
                overlay.snap = line;
                if !*dup {
                    overlay.ghost = picks.iter().map(|&(t, i)| (t, drag.snapshot.tracks[t].clips[i].start, drag.snapshot.tracks[t].clips[i].len)).collect();
                }
                // The record viewer shows the dragged clip's head.
                let hc = &drag.snapshot.tracks[head.0].clips[head.1];
                self.rec.peek = Some(hc.start + dt);
                let wv = picks.iter().any(|(t, _)| model::VIDEO_TRACKS.contains(t));
                let ht = model::moved_track(head.0, dtr, wv);
                self.rec.sel = match &drag.sel {
                    Selection::Clips(v) => Selection::Clips(v.iter().map(|&(t, st)| (model::moved_track(t, dtr, wv), st + dt)).collect()),
                    _ => Selection::Clip { track: ht, start: hc.start + dt },
                };
            }
            Kind::Trim { track, at, side, ripple } => {
                let (s, line) = self.snap(&[at + d], &[*at], &drag.snapshot);
                if let Ok(moved) = seq.trim(*track, *at, *side, d + s, *ripple) {
                    let at2 = if *side == Side::In && *ripple { *at } else { at + moved };
                    // The trim monitor follows the edit.
                    self.rec.sel = Selection::Edit { track: *track, at: at2, side: *side };
                    overlay.snap = line.filter(|_| moved == d + s);
                    // The source the trimmed clip(s) have past their edges.
                    let (a, b) = seq.tracks[*track].cut(at2);
                    let sides: &[Option<usize>] = match side {
                        Side::Out => &[a],
                        Side::In => &[b],
                        Side::Roll => &[a, b],
                    };
                    for i in sides.iter().flatten() {
                        let c = &seq.tracks[*track].clips[*i];
                        let (h0, h1) = seq.handles(c);
                        overlay.handles.push((*track, c.start - h0, c.start, c.end(), c.end() + h1));
                    }
                }
            }
            Kind::Stretch { track, idx, side } => {
                let c = &drag.snapshot.tracks[*track].clips[*idx];
                let edge = if *side == Side::In { c.start } else { c.end() };
                let (s, line) = self.snap(&[edge + d], &[edge], &drag.snapshot);
                let len = seq.stretch(*track, *idx, *side, d + s);
                overlay.snap = line;
                let nc = &seq.tracks[*track].clips[*idx];
                self.rec.sel = Selection::Clip { track: *track, start: nc.start };
                self.rec.peek = Some(if *side == Side::In { nc.start } else { nc.end() - 1 });
                self.notice = Some((format!("rate stretch: {len} frames at {:.3}x", nc.speed), std::time::Instant::now()));
            }
            Kind::Slip { track, idx } => {
                // The picture moves with the hand: drag right shows earlier source.
                let c = &drag.snapshot.tracks[*track].clips[*idx];
                let src_d = -c.src_offset(d, drag.snapshot.fps, drag.snapshot.media[c.media].fps);
                let moved = seq.slip(*track, *idx, src_d);
                let nc = &seq.tracks[*track].clips[*idx];
                self.rec.peek = Some(nc.start);
                self.slip_view = Some((*track, nc.start, std::time::Instant::now()));
                let (h0, h1) = seq.handles(nc);
                overlay.handles.push((*track, nc.start - h0, nc.start, nc.end(), nc.end() + h1));
                self.notice = Some((format!("slip {moved:+} source frames: in {}", nc.src_in), std::time::Instant::now()));
            }
            Kind::Speed { track, idx, anchor } => {
                // Right: faster (twice per 120 px), left: slower.
                let c = &drag.snapshot.tracks[*track].clips[*idx];
                let px = df / self.rec.view.fpp().max(1e-9);
                let speed = (c.speed * 2f64.powf(px / 120.0)).clamp(-crate::transport::MAX_SPEED, crate::transport::MAX_SPEED);
                let speed = (speed * 1000.0).round() / 1000.0;
                let (st, len) = seq.speed_about(*track, *idx, if speed == 0.0 { c.speed } else { speed }, *anchor);
                self.rec.sel = Selection::Clip { track: *track, start: st };
                self.rec.peek = Some((*anchor).clamp(st, st + len - 1));
                self.notice = Some((format!("speed {speed:.3}x: {len} frames, about {anchor}"), std::time::Instant::now()));
            }
            Kind::Slide { track, idx } => {
                let c = &drag.snapshot.tracks[*track].clips[*idx];
                let (s, line) = self.snap(&[c.start + d, c.end() + d], &[c.start, c.end()], &drag.snapshot);
                let moved = seq.slide(*track, *idx, d + s);
                overlay.snap = line.filter(|_| moved == d + s);
                let nc = &seq.tracks[*track].clips[*idx];
                self.rec.sel = Selection::Clip { track: *track, start: nc.start };
                self.rec.peek = Some(nc.start);
                self.notice = Some((format!("slide {moved:+} frames"), std::time::Instant::now()));
            }
            Kind::Dissolve { track, at, end } => {
                if let Some(x) = drag.snapshot.tracks[*track].dissolve_at(*at).copied() {
                    let edge = if *end { x.end() } else { x.start() };
                    let (s, line) = self.snap(&[edge + d], &[edge], &drag.snapshot);
                    let (before, after) = if *end { (x.before, x.after + d + s) } else { (x.before - d - s, x.after) };
                    let _ = seq.set_dissolve_sides(*track, *at, before, after);
                    overlay.snap = line;
                }
            }
        }
        self.rec.seq = seq;
        self.rec.fit_transport();
        self.overlay = overlay;
        self.drag = Some(drag);
    }

    fn drag_end(&mut self) {
        let Some(drag) = self.drag.take() else { return };
        self.overlay = Overlay::default();
        self.rec.peek = None;
        if self.rec.seq != drag.snapshot {
            // One undo step: the history's baseline is still the snapshot.
            self.changed(false);
        }
    }

    /// Esc during a drag: everything back as it was at the press.
    pub(crate) fn drag_cancel(&mut self) -> bool {
        let Some(drag) = self.drag.take() else { return false };
        self.rec.seq = drag.snapshot;
        self.rec.sel = drag.sel;
        self.rec.peek = None;
        self.overlay = Overlay::default();
        self.rec.fit_transport();
        self.drag_cancelled = true;
        true
    }

    pub fn dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// The screen point over `frame` in `track`'s lane (headless mouse tests).
    pub fn timeline_point(&self, track: usize, frame: f64) -> Option<egui::Pos2> {
        Some(egui::pos2(self.rec.view.frame_x(frame), self.rec.view.track_y(track)?))
    }

    /// Undo once (headless tests).
    pub fn undo_once(&mut self) {
        self.undo_public();
    }
}
