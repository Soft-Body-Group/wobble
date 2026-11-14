//! The timeline panel: a ruler, V3 V2 V1 A1 A2 A3 M (top to bottom) with
//! their clips, the in/out range and the playhead. Track headers show and
//! set the edit targets (click the name); audio headers also carry mute,
//! solo and the track's gain (drag it; double-click resets).
//!
//! Hands: the TOOL (strip at the top left, or V R N Y U) decides what a
//! press on a clip does, unless a HOLD key is down (A S D Z C, shift A S
//! D: see `mouse::HoldOp`), which decides instead, anywhere on the clip. LMB on the ruler scrubs; a click on
//! an empty lane moves the playhead and clears the selection; a drag from
//! an empty lane draws a marquee. Right-click: the menu for what is under
//! the pointer (dissolves, razor, delete, speed). The wheel zooms about the
//! pointer, shift+wheel (or a trackpad sideways) pans; MMB drag pans. Clips
//! and the cross dissolve can be dropped on from the source viewer and the
//! bin (`Payload`).
//!
//! Buttons laid over the panel (tools, M / S, the ruler's chip) sense drags
//! too: the panel itself is drag-sensitive, and egui hands a press to the
//! drag widget underneath a click-only one.

use egui::{pos2, vec2, Align2, Color32, FontId, PointerButton, Rect, Sense, Stroke, Ui};
use imfram::timeline::{fit_view, frame_to_px, px_to_frame, ruler_step, zoom_view};
use model::{Kind, Marks, Sequence, Targets, A1, A2, A3, M, V1, V2, V3};

const HEADER_W: f32 = 132.0;
const RULER_H: f32 = 20.0;
/// Track rows grow to this when there is room, and shrink to `TRACK_MIN`.
const TRACK_H: f32 = 34.0;
const TRACK_MIN: f32 = 18.0;
const GAP: f32 = 2.0;
const LABEL_FONT: f32 = 10.0;
const CLIP_FONT: f32 = 11.0;
/// Track rows top to bottom.
const ROWS: [usize; 7] = [V3, V2, V1, A1, A2, A3, M];

/// What a press on the tracks does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Tool {
    /// Select, move, trim (V).
    #[default]
    Select,
    /// Click cuts the clip under the pointer (shift: every track) (X).
    Razor,
    /// Drag an edge: the clip's speed changes to fill the new length (no
    /// key: hold shift+A / shift+D and drag; the sequence menu has it).
    Rate,
    /// Drag a clip: other source frames in the same place (Y).
    Slip,
    /// Drag a clip: it moves, its neighbours give and take (U).
    Slide,
    /// Drag a cut: it moves, one clip longer and the other shorter (N).
    Roll,
}

impl Tool {
    pub const ALL: [Tool; 6] = [Tool::Select, Tool::Razor, Tool::Roll, Tool::Rate, Tool::Slip, Tool::Slide];

    /// Its hotkey ("" for the rate tool: shift A / shift D hold-drags do
    /// it now; the sequence menu still has the tool).
    pub fn key(self) -> &'static str {
        match self {
            Tool::Select => "V",
            Tool::Razor => "X",
            Tool::Rate => "",
            Tool::Slip => "Y",
            Tool::Slide => "U",
            Tool::Roll => "N",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Tool::Select => "select",
            Tool::Razor => "razor",
            Tool::Rate => "rate stretch",
            Tool::Slip => "slip",
            Tool::Slide => "slide",
            Tool::Roll => "roll",
        }
    }
}

/// What can be dragged onto the timeline (egui drag and drop).
#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    /// The source viewer's clip, its marks (`len` record frames).
    Source { len: i64 },
    /// A file from the bin (the whole clip).
    File(std::path::PathBuf),
    /// Several files (rows dragged out of the clip browser): laid down one
    /// after another, in order.
    #[cfg_attr(not(feature = "clips"), allow(dead_code))]
    Files(Vec<std::path::PathBuf>),
    /// The cross dissolve, for a cut.
    Dissolve,
    /// A new text clip (the bin's generator).
    Text,
}

/// A command from the timeline's right-click menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmd {
    AddDissolve { track: usize, at: i64 },
    RemoveDissolve { track: usize, at: i64 },
    SelectEdit { track: usize, at: i64 },
    Razor { track: usize, at: i64, all: bool },
    Lift { track: usize, start: i64 },
    RippleDelete { track: usize, start: i64 },
    Speed { track: usize, start: i64 },
    /// A clip's file read as this colourspace (`None`: by its format).
    Input { media: usize, space: Option<colour::color::InputSpace> },
}

/// A change to a track's sound from its header.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackOp {
    Mute,
    Solo,
    /// The gain in dB, as dragged (`done`: the drag ended).
    Gain(f32, bool),
}

/// What is selected on the record side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    None,
    /// A clip (with its linked group), by track and start.
    Clip { track: usize, start: i64 },
    /// Several clips (shift / cmd click), each by track and start.
    Clips(Vec<(usize, i64)>),
    /// An edit point: a cut on a track and which side trims move.
    Edit { track: usize, at: i64, side: model::Side },
}

pub struct TimelineView {
    /// (frame at the left edge, frames per pixel); scale <= 0: fit next draw.
    pub view: (f64, f64),
    strip: Rect,
    rect: Rect,
    /// The press a drag started from: what it hit, where (frame, y).
    press: Option<(Hit, f64, f32)>,
    /// The modifiers on the press (held through the drag with the live ones).
    press_mods: egui::Modifiers,
    /// The height of a track row this frame.
    track_h: f32,
    /// A marquee being drawn: where it started (screen).
    marquee: Option<egui::Pos2>,
    /// What the right-click menu was opened on: hit, frame, row's track.
    menu: Option<(Hit, i64, Option<usize>)>,
    /// A navigation drag is running (it is not a scrub, marquee or edit).
    navving: bool,
    /// The view that shows the whole sequence, as of the last draw.
    fit_at: (f64, f64),
}

pub enum Event {
    None,
    Scrub(i64),
    Targets(Targets),
    Track(usize, TrackOp),
    /// A click (no drag) on something (or an empty lane); `add`: shift or
    /// cmd held; `frame` under the pointer; the modifiers.
    Click { hit: Hit, add: bool, frame: f64, mods: egui::Modifiers },
    /// A marquee over these clips (track, start); `add`: shift held.
    Marquee { picks: Vec<(usize, i64)>, add: bool, done: bool },
    /// Something dropped at `frame` on a row.
    Drop { payload: Payload, frame: i64, track: Option<usize>, insert: bool },
    Menu(Cmd),
    Tool(Tool),
    /// The ruler's "× in/out" chip.
    ClearMarks,
    /// A double-click on a clip.
    Open { track: usize, start: i64 },
    /// A drag from `hit`: frames moved since the press (unsnapped), rows
    /// moved (+ = up, toward V3), the modifiers now, and whether it ended.
    Drag { hit: Hit, df: f64, dtrack: i32, mods: egui::Modifiers, done: bool },
}

/// What a press landed on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    /// A clip's body, by track and start.
    Clip { track: usize, start: i64 },
    /// A clip's edge: `Out` = the end of the clip ending at `at`, `In` =
    /// the start of the clip starting there.
    Edge { track: usize, at: i64, side: model::Side },
    /// A dissolve's start (`end` false) or end edge.
    Dissolve { track: usize, at: i64, end: bool },
    /// Empty lane or the ruler.
    Empty,
}

/// What the editor wants drawn over the tracks while a drag runs.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    /// Where the dragged clips were: (track, start, len), drawn as outlines.
    pub ghost: Vec<(usize, i64, i64)>,
    /// A snap line at this frame.
    pub snap: Option<i64>,
    /// The source a trimmed clip has beyond its edges: (track, first frame
    /// it could start on, start, end, last frame it could end on).
    pub handles: Vec<(usize, i64, i64, i64, i64)>,
}

/// Pixels a press may miss an edge by.
const EDGE_PX: f32 = 5.0;

impl TimelineView {
    pub fn new() -> Self {
        Self { view: (0.0, 0.0), strip: Rect::NOTHING, rect: Rect::NOTHING, press: None, press_mods: egui::Modifiers::NONE, track_h: TRACK_H, marquee: None, menu: None, navving: false, fit_at: (0.0, 0.0) }
    }

    /// Frames per pixel (snapping distances).
    pub fn fpp(&self) -> f64 {
        self.view.1
    }

    /// Screen x of frame `f` and the vertical middle of `track`'s lane
    /// (headless mouse tests aim with these).
    pub fn frame_x(&self, f: f64) -> f32 {
        frame_to_px(self.view, self.strip.min.x, f)
    }

    pub fn track_y(&self, track: usize) -> Option<f32> {
        let i = ROWS.iter().position(|&t| t == track)?;
        Some(self.rect.min.y + RULER_H + GAP + i as f32 * (self.track_h + GAP) + self.track_h / 2.0)
    }

    /// The track whose lane is at screen height `y`.
    fn row_at(&self, y: f32) -> Option<usize> {
        (0..ROWS.len())
            .find(|&i| {
                let top = self.rect.min.y + RULER_H + GAP + i as f32 * (self.track_h + GAP);
                y >= top && y < top + self.track_h
            })
            .map(|i| ROWS[i])
    }

    /// What is at a screen position.
    fn hit(&self, seq: &Sequence, q: egui::Pos2) -> Hit {
        if q.x < self.strip.min.x || q.y < self.rect.min.y + RULER_H {
            return Hit::Empty;
        }
        let Some(i) = (0..ROWS.len()).find(|&i| {
            let y = self.rect.min.y + RULER_H + GAP + i as f32 * (self.track_h + GAP);
            q.y >= y && q.y < y + self.track_h
        }) else {
            return Hit::Empty;
        };
        let ti = ROWS[i];
        let fx = |f: i64| frame_to_px(self.view, self.strip.min.x, f as f64);
        let t = &seq.tracks[ti];
        for d in &t.dissolves {
            if (q.x - fx(d.start())).abs() <= EDGE_PX {
                return Hit::Dissolve { track: ti, at: d.at, end: false };
            }
            if (q.x - fx(d.end())).abs() <= EDGE_PX {
                return Hit::Dissolve { track: ti, at: d.at, end: true };
            }
        }
        // Edges: at a butt cut the side of the cut the press is on decides
        // (left: the outgoing clip's out, right: the incoming clip's in);
        // an edge with nothing beside it can be caught a few pixels outside.
        for (k, c) in t.clips.iter().enumerate() {
            let (x0, x1) = (fx(c.start), fx(c.end()));
            let butt_before = k > 0 && t.clips[k - 1].end() == c.start;
            let butt_after = t.clips.get(k + 1).is_some_and(|n| n.start == c.end());
            let room = ((x1 - x0) / 3.0).min(EDGE_PX).max(2.0);
            let out_reach = if butt_after { 0.0 } else { EDGE_PX };
            let in_reach = if butt_before { 0.0 } else { EDGE_PX };
            if q.x >= x1 - room && q.x < x1 + out_reach {
                return Hit::Edge { track: ti, at: c.end(), side: model::Side::Out };
            }
            if q.x >= x0 - in_reach && q.x < x0 + room {
                return Hit::Edge { track: ti, at: c.start, side: model::Side::In };
            }
            if q.x >= x0 && q.x < x1 {
                return Hit::Clip { track: ti, start: c.start };
            }
        }
        Hit::Empty
    }

    pub fn fit(&mut self) {
        self.view.1 = 0.0;
    }

    /// Showing the whole sequence (R's fit).
    pub fn fitted(&self) -> bool {
        self.view.1 <= 0.0 || ((self.view.0 - self.fit_at.0).abs() < 1e-6 && (self.view.1 - self.fit_at.1).abs() < 1e-9)
    }

    pub fn zoom(&mut self, factor: f32) {
        if self.strip.is_positive() {
            self.view = zoom_view(self.view, self.strip.width() * 0.5, factor);
        }
    }

    /// Keep frame `t` on screen (playback runs off the right edge).
    pub fn follow(&mut self, t: i64) {
        if !self.strip.is_positive() || self.view.1 <= 0.0 {
            return;
        }
        let w = self.strip.width() as f64 * self.view.1;
        let t = t as f64;
        if t < self.view.0 || t > self.view.0 + w {
            self.view.0 = t - w * 0.1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ui(&mut self, ui: &mut Ui, seq: &Sequence, playhead: i64, marks: Marks, targets: Targets, focused: bool, sel: &Selection, ripple: bool, overlay: &Overlay, tool: Tool, hold: Option<&'static str>) -> Event {
        let sel = sel.clone();
        let mut ev = Event::None;
        let avail = ui.available_rect_before_wrap();
        let (rect, r) = ui.allocate_exact_size(avail.size(), Sense::click_and_drag());
        let strip = Rect::from_min_max(pos2(rect.min.x + HEADER_W, rect.min.y), rect.max);
        self.strip = strip;
        self.rect = rect;
        self.track_h = ((rect.height() - RULER_H - GAP) / ROWS.len() as f32 - GAP).clamp(TRACK_MIN, TRACK_H);
        let track_h = self.track_h;
        let len = seq.duration().max(240);
        self.fit_at = fit_view((0, len), strip.width());
        if self.view.1 <= 0.0 {
            self.view = self.fit_at;
        }
        let v = ui.visuals().clone();
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, v.extreme_bg_color);
        if focused {
            p.rect_stroke(rect.shrink(0.5), 0.0, Stroke::new(1.0, v.selection.bg_fill), egui::StrokeKind::Inside);
        }
        // Navigation.
        if r.hovered() {
            let w = imfram::nav::wheel_nav(ui);
            if w.zoom != 1.0 {
                let x = r.hover_pos().map_or(strip.center().x, |p| p.x) - strip.min.x;
                self.view = zoom_view(self.view, x, w.zoom);
            }
            self.view.0 -= w.pan.x as f64 * self.view.1;
        }
        // Drag navigation, imfram's (wobble's) table: MMB pans, ctrl+MMB
        // drag-zooms, cmd/alt+RMB drag-zooms, alt+LMB pans from an empty
        // lane or the ruler (on a clip alt+LMB duplicates, Premiere's).
        let mods_now = ui.input(|i| i.modifiers);
        let press_on_clip = ui.input(|i| i.pointer.press_origin()).is_some_and(|q| q.y >= rect.min.y + RULER_H && self.hit(seq, q) != Hit::Empty);
        for b in [PointerButton::Middle, PointerButton::Primary, PointerButton::Secondary] {
            if !r.dragged_by(b) || (b == PointerButton::Primary && (press_on_clip || self.press.is_some_and(|p| p.0 != Hit::Empty))) {
                continue;
            }
            match imfram::nav::drag_nav(b, mods_now) {
                Some(imfram::nav::DragNav::Pan) => {
                    self.view.0 -= r.drag_delta().x as f64 * self.view.1;
                    self.navving = true;
                }
                Some(imfram::nav::DragNav::Zoom) => {
                    let x = ui.input(|i| i.pointer.press_origin()).map_or(strip.center().x, |p| p.x) - strip.min.x;
                    self.view = zoom_view(self.view, x, imfram::nav::drag_zoom(r.drag_delta()));
                    self.navving = true;
                }
                None => {}
            }
        }
        if !ui.input(|i| i.pointer.any_down()) {
            self.navving = false;
        }

        let view = self.view;
        let fx = |f: f64| frame_to_px(view, strip.min.x, f);

        // Rows.
        let row_rect = |i: usize| {
            let y = rect.min.y + RULER_H + GAP + i as f32 * (track_h + GAP);
            Rect::from_min_size(pos2(rect.min.x, y), vec2(rect.width(), track_h))
        };
        let ruler = Rect::from_min_max(pos2(strip.min.x, rect.min.y), pos2(strip.max.x, rect.min.y + RULER_H));
        p.rect_filled(ruler, 0.0, v.widgets.noninteractive.bg_fill);
        // The tools, in the corner over the headers.
        let bw = (HEADER_W - GAP * 2.0) / Tool::ALL.len() as f32;
        for (k, t) in Tool::ALL.into_iter().enumerate() {
            let b = Rect::from_min_size(pos2(rect.min.x + GAP + k as f32 * bw, rect.min.y + 1.0), vec2(bw - 2.0, RULER_H - 2.0));
            let k = if t.key().is_empty() { "↔" } else { t.key() };
            let label = egui::RichText::new(k).size(11.0).strong();
            let tip = if t.key().is_empty() { format!("{} tool (hold shift A / shift D and drag does it too)", t.name()) } else { format!("{} tool ({})", t.name(), t.key()) };
            let r = ui.put(b, egui::Button::selectable(tool == t, label).sense(Sense::click_and_drag())).on_hover_text(tip);
            if r.clicked() {
                ev = Event::Tool(t);
            }
        }

        // In/out range under everything.
        if marks.inp.is_some() || marks.out.is_some() {
            let a = marks.inp.unwrap_or(0) as f64;
            let b = marks.out.map_or(len as f64, |o| o as f64 + 1.0);
            let lit = Rect::from_x_y_ranges(fx(a)..=fx(b), rect.y_range()).intersect(strip);
            p.rect_filled(lit, 0.0, v.selection.bg_fill.linear_multiply(0.25));
        }

        // The ruler's "× clear in/out" chip at the marked range's end.
        if marks.inp.is_some() || marks.out.is_some() {
            let b = marks.out.map_or(len as f64, |o| o as f64 + 1.0);
            let x = if strip.width() > 112.0 { fx(b).clamp(strip.min.x + 56.0, strip.max.x - 56.0) } else { strip.center().x };
            let chip = Rect::from_center_size(pos2(x, ruler.center().y), vec2(104.0, RULER_H - 2.0));
            let cr = ui
                .put(chip, egui::Button::new(egui::RichText::new("× clear in/out").size(11.0).strong().color(egui::Color32::WHITE)).fill(crate::editor::CLEAR_MARKS).small().sense(Sense::click_and_drag()))
                .on_hover_text("clear the record in and out (alt X)");
            if cr.clicked() {
                ev = Event::ClearMarks;
            }
        }

        // Ruler ticks.
        let step = ruler_step(self.view.1);
        let minor = if step >= 5 { step / 5 } else { 1 };
        let first = (px_to_frame(self.view, strip.min.x, strip.min.x) / minor as f64).floor() as i64 * minor;
        let last = px_to_frame(self.view, strip.min.x, strip.max.x).ceil() as i64;
        let tick = v.widgets.noninteractive.fg_stroke.color;
        let mut f = first.max(0);
        while f <= last {
            let x = fx(f as f64);
            let major = f.rem_euclid(step) == 0;
            let h = if major { RULER_H * 0.5 } else { RULER_H * 0.25 };
            p.line_segment([pos2(x, ruler.max.y - h), pos2(x, ruler.max.y)], Stroke::new(1.0, tick));
            if major {
                let tc = crate::editor::timecode(f as u64, seq.fps.as_f64());
                p.text(pos2(x + 3.0, ruler.min.y + 1.0), Align2::LEFT_TOP, tc, FontId::proportional(LABEL_FONT), v.weak_text_color());
            }
            f += minor;
        }

        let pointer = ui.input(|i| i.pointer.interact_pos());
        let clicked = r.clicked_by(PointerButton::Primary);
        for (i, &ti) in ROWS.iter().enumerate() {
            let row = row_rect(i);
            let track = &seq.tracks[ti];
            let header = Rect::from_min_max(row.min, pos2(strip.min.x - GAP, row.max.y));
            let targeted = targets.has(ti);
            let hfill = if targeted { v.selection.bg_fill } else { v.widgets.inactive.bg_fill };
            p.rect_filled(header, 2.0, hfill);
            let key = match ti {
                V1 => "1",
                V2 => "2",
                V3 => "3",
                A1 => "4",
                A2 => "5",
                A3 => "6",
                _ => "7",
            };
            let name_font = FontId::proportional(if track_h < 24.0 { 11.0 } else { 13.0 });
            p.text(header.left_center() + vec2(6.0, 0.0), Align2::LEFT_CENTER, track.name, name_font, v.strong_text_color());
            p.text(header.left_center() + vec2(30.0, 0.0), Align2::LEFT_CENTER, key, FontId::monospace(10.0), v.weak_text_color());
            // The name end of the header toggles the target.
            let name_box = Rect::from_min_max(header.min, pos2(header.min.x + 44.0, header.max.y));
            if clicked && pointer.is_some_and(|q| name_box.contains(q)) {
                let mut t = targets;
                t.toggle(ti);
                ev = Event::Targets(t);
            }
            if track.kind == Kind::Audio {
                let bh = (track_h - 6.0).clamp(12.0, 18.0);
                let at = |x: f32, w: f32| Rect::from_center_size(pos2(header.min.x + x + w / 2.0, header.center().y), vec2(w, bh));
                let small = |s: &str| egui::RichText::new(s).size(10.0);
                let mute = ui.put(at(46.0, 18.0), egui::Button::selectable(track.mute, small("M")).min_size(vec2(18.0, bh)).sense(Sense::click_and_drag()));
                if mute.clicked() {
                    ev = Event::Track(ti, TrackOp::Mute);
                }
                let solo = ui.put(at(66.0, 18.0), egui::Button::selectable(track.solo, small("S")).min_size(vec2(18.0, bh)).sense(Sense::click_and_drag()));
                if solo.clicked() {
                    ev = Event::Track(ti, TrackOp::Solo);
                }
                let mut db = track.gain_db;
                let g = ui.put(at(88.0, header.max.x - header.min.x - 92.0), egui::DragValue::new(&mut db).range(-60.0..=12.0).speed(0.2).max_decimals(1).custom_formatter(|d, _| if d.abs() < 0.05 { "0 dB".into() } else { format!("{d:+.1}") }));
                let g = g.on_hover_text(format!("{} gain: drag (double-click: 0 dB)", track.name));
                if g.double_clicked() {
                    ev = Event::Track(ti, TrackOp::Gain(0.0, true));
                } else if g.changed() || g.drag_stopped() {
                    ev = Event::Track(ti, TrackOp::Gain(db, g.drag_stopped() || g.lost_focus()));
                }
                if track.mute || (seq.tracks.iter().any(|t| t.solo) && !track.solo) {
                    p.rect_filled(Rect::from_min_max(pos2(strip.min.x, row.min.y), row.max), 0.0, Color32::from_black_alpha(60));
                }
            }
            let lane = Rect::from_min_max(pos2(strip.min.x, row.min.y), row.max);
            p.rect_filled(lane, 0.0, v.faint_bg_color);
            let base = match track.kind {
                Kind::Video => Color32::from_rgb(60, 110, 160),
                Kind::Audio => Color32::from_rgb(70, 140, 100),
            };
            for c in &track.clips {
                let x0 = fx(c.start as f64).max(strip.min.x - 2.0);
                let x1 = fx(c.end() as f64).min(strip.max.x + 2.0);
                if x1 < strip.min.x || x0 > strip.max.x {
                    continue;
                }
                let cr = Rect::from_min_max(pos2(x0, lane.min.y + 1.0), pos2(x1, lane.max.y - 1.0));
                let base = if ti == M {
                    Color32::from_rgb(130, 100, 160)
                } else if c.text.is_some() {
                    Color32::from_rgb(170, 130, 40)
                } else {
                    base
                };
                let under = cr.contains(pos2(fx(playhead as f64 + 0.5), cr.center().y));
                let fill = if under { base.gamma_multiply(1.25) } else { base };
                p.rect_filled(cr, 2.0, fill);
                let picked = |st: usize, start: i64| {
                    let sc = seq.tracks[st].clips.iter().find(|x| x.start == start);
                    sc.is_some_and(|sc| sc.link == c.link && sc.start == c.start)
                };
                let selected = match &sel {
                    Selection::Clip { track: st, start } => picked(*st, *start),
                    Selection::Clips(v) => v.iter().any(|&(st, start)| picked(st, start)),
                    _ => false,
                };
                // A Wobble clip whose active version is not rendered plays
                // its source: drawn red until the render lands (rule 8).
                let unrendered = c.wobble.as_ref().is_some_and(|w| c.media == w.source);
                let edge = if selected {
                    Stroke::new(2.0, Color32::WHITE)
                } else if unrendered {
                    Stroke::new(2.0, Color32::from_rgb(230, 60, 60))
                } else if c.wobble.is_some() || c.wob.is_some() {
                    Stroke::new(1.5, Color32::from_rgb(120, 220, 230))
                } else {
                    Stroke::new(1.0, Color32::from_black_alpha(140))
                };
                p.rect_stroke(cr, 2.0, edge, egui::StrokeKind::Inside);
                if cr.width() > 24.0 && track_h >= 24.0 {
                    let cp = p.with_clip_rect(cr.shrink(2.0).intersect(strip));
                    let m = &seq.media[c.media];
                    let label = match (&c.text, &c.wobble) {
                        (Some(t), _) => format!("T  {}", t.text.lines().next().unwrap_or("")),
                        (_, Some(w)) => format!("W v{:03}  {}", w.active, seq.media[w.source].name),
                        _ if c.wob.is_some() => format!("WOB  {}", m.name),
                        _ => m.name.clone(),
                    };
                    cp.text(cr.left_top() + vec2(4.0, 3.0), Align2::LEFT_TOP, label, FontId::proportional(CLIP_FONT), Color32::WHITE);
                    let mut tc = if c.speed == 1.0 { format!("{}  +{}", c.src_in, c.len) } else { format!("{}  +{}  {}x", c.src_in, c.len, c.speed) };
                    if c.conform {
                        tc.push_str("  conf");
                    }
                    cp.text(cr.left_bottom() + vec2(4.0, -3.0), Align2::LEFT_BOTTOM, tc, FontId::monospace(9.0), Color32::from_white_alpha(170));
                }
            }
        }

        // Dissolves: a bright band across the cut, the fade drawn as a
        // ramp, its length written on it when there is room.
        for (i, &ti) in ROWS.iter().enumerate() {
            let lane = row_rect(i);
            for d in &seq.tracks[ti].dissolves {
                let full = Rect::from_x_y_ranges(fx(d.start() as f64)..=fx(d.end() as f64), (lane.min.y + 2.0)..=(lane.max.y - 2.0));
                let r = full.intersect(strip);
                if r.width() <= 0.0 {
                    continue;
                }
                let dp = p.with_clip_rect(r);
                dp.rect_filled(full, 3.0, Color32::from_rgba_unmultiplied(235, 235, 250, 200));
                let ramp = Color32::from_rgb(70, 70, 110);
                dp.add(egui::Shape::convex_polygon(vec![full.left_bottom(), full.right_top(), full.right_bottom()], ramp.gamma_multiply(0.55), Stroke::NONE));
                dp.line_segment([full.left_bottom(), full.right_top()], Stroke::new(1.5, ramp));
                p.rect_stroke(r, 3.0, Stroke::new(1.0, Color32::from_rgb(40, 40, 60)), egui::StrokeKind::Inside);
                if full.width() > 34.0 && track_h >= 20.0 {
                    dp.text(full.left_top() + vec2(3.0, 1.0), Align2::LEFT_TOP, format!("× {}", d.len()), FontId::proportional(9.5), Color32::from_rgb(30, 30, 50));
                }
            }
        }

        // The selected edit point: a bracket on its side of the cut.
        if let Selection::Edit { track, at, side } = sel.clone() {
            if let Some(i) = ROWS.iter().position(|&t| t == track) {
                let lane = row_rect(i);
                let x = fx(at as f64);
                let col = if ripple { Color32::from_rgb(240, 210, 60) } else { Color32::from_rgb(230, 60, 60) };
                let st = Stroke::new(3.0, col);
                let (y0, y1) = (lane.min.y, lane.max.y);
                let w = 7.0;
                let bracket = |dir: f32| {
                    p.line_segment([pos2(x, y0), pos2(x, y1)], st);
                    p.line_segment([pos2(x, y0), pos2(x + dir * w, y0)], st);
                    p.line_segment([pos2(x, y1), pos2(x + dir * w, y1)], st);
                };
                match side {
                    model::Side::Out => bracket(-1.0),
                    model::Side::In => bracket(1.0),
                    model::Side::Roll => {
                        bracket(-1.0);
                        bracket(1.0);
                    }
                }
            }
        }

        // While trimming: the source the clip has past its edges.
        for &(ti, first, start, end, last) in &overlay.handles {
            if let Some(i) = ROWS.iter().position(|&t| t == ti) {
                let lane = row_rect(i);
                let band = Color32::from_rgba_unmultiplied(120, 200, 255, 40);
                for (k, (a, b)) in [(first, start), (end, last)].into_iter().enumerate() {
                    if b <= a {
                        // At the limit on this side: say so by the edge.
                        let (x, al) = if k == 0 { (fx(start as f64) - 4.0, Align2::RIGHT_CENTER) } else { (fx(end as f64) + 4.0, Align2::LEFT_CENTER) };
                        p.text(pos2(x, lane.center().y), al, "no handle", FontId::proportional(9.5), Color32::from_rgb(255, 150, 150));
                        continue;
                    }
                    let hr = Rect::from_x_y_ranges(fx(a as f64)..=fx(b as f64), (lane.min.y + 3.0)..=(lane.max.y - 3.0)).intersect(strip);
                    if hr.width() <= 0.0 {
                        continue;
                    }
                    p.rect_filled(hr, 2.0, band);
                    p.rect_stroke(hr, 2.0, Stroke::new(1.0, Color32::from_rgb(120, 200, 255)), egui::StrokeKind::Inside);
                    if hr.width() > 30.0 {
                        p.text(hr.center(), Align2::CENTER_CENTER, format!("+{}", b - a), FontId::monospace(9.0), Color32::from_rgb(170, 225, 255));
                    }
                }
            }
        }

        // Where the dragged clips were, and the snap line.
        for &(ti, start, len) in &overlay.ghost {
            if let Some(i) = ROWS.iter().position(|&t| t == ti) {
                let lane = row_rect(i);
                let gr = Rect::from_min_max(pos2(fx(start as f64), lane.min.y + 1.0), pos2(fx((start + len) as f64), lane.max.y - 1.0)).intersect(strip);
                if gr.width() > 0.0 {
                    p.rect_stroke(gr, 2.0, Stroke::new(1.0, Color32::from_white_alpha(110)), egui::StrokeKind::Inside);
                }
            }
        }
        if let Some(f) = overlay.snap {
            let x = fx(f as f64);
            p.line_segment([pos2(x, rect.min.y), pos2(x, rect.max.y)], Stroke::new(1.0, Color32::from_rgb(240, 210, 60)));
        }

        // The mouse: clicks select (or cut, with the razor), drags move /
        // trim / slip / slide / stretch, the ruler scrubs, an empty lane
        // clicks to the playhead or draws a marquee.
        let in_strip = |q: egui::Pos2| q.x >= strip.min.x;
        let in_ruler = |q: egui::Pos2| q.y < rect.min.y + RULER_H;
        // Modifiers on this frame's button events count as held (a press
        // with cmd, or a test injecting one), with the live state.
        let evmods = ui.input(|i| {
            i.events.iter().fold(egui::Modifiers::NONE, |m, e| match e {
                egui::Event::PointerButton { modifiers, pressed: true, .. } => m | *modifiers,
                _ => m,
            })
        });
        if evmods != egui::Modifiers::NONE {
            self.press_mods = evmods;
        }
        let mods = ui.input(|i| i.modifiers) | self.press_mods;
        let hover = ui.input(|i| i.pointer.hover_pos()).filter(|q| rect.contains(*q));
        if let Some(q) = hover {
            let under = self.press.map(|p| p.0).unwrap_or_else(|| self.hit(seq, q));
            // What a press on an edge would do, named by the cursor and a
            // label by the pointer: trim in / trim out (one-sided arrows),
            // ripple (the two-way arrow), roll (the column), rate stretch.
            let edge_mode = match (tool, under) {
                // A held key decides, anywhere on a clip.
                (_, Hit::Clip { .. } | Hit::Edge { .. }) if hold.is_some() => hold.map(|w| (w, egui::CursorIcon::ResizeHorizontal)),
                (Tool::Roll, Hit::Edge { .. } | Hit::Dissolve { .. }) => Some(("roll", egui::CursorIcon::ResizeColumn)),
                (Tool::Select, Hit::Edge { side, .. }) if ripple || mods.command => {
                    Some((if side == model::Side::In { "ripple in" } else { "ripple out" }, egui::CursorIcon::ResizeHorizontal))
                }
                (Tool::Select, Hit::Edge { side: model::Side::In, .. }) => Some(("trim in", egui::CursorIcon::ResizeEast)),
                (Tool::Select, Hit::Edge { .. }) => Some(("trim out", egui::CursorIcon::ResizeWest)),
                (Tool::Select, Hit::Dissolve { .. }) => Some(("dissolve length", egui::CursorIcon::ResizeHorizontal)),
                (Tool::Rate, Hit::Edge { .. }) => Some(("rate stretch", egui::CursorIcon::ResizeHorizontal)),
                (Tool::Slip, Hit::Clip { .. } | Hit::Edge { .. }) => Some(("slip", egui::CursorIcon::AllScroll)),
                (Tool::Slide, Hit::Clip { .. } | Hit::Edge { .. }) => Some(("slide", egui::CursorIcon::Move)),
                _ => None,
            };
            if let Some((word, _)) = edge_mode.filter(|_| !(tool == Tool::Razor && hold.is_none())) {
                let col = if word.starts_with("ripple") { Color32::from_rgb(240, 210, 60) } else { Color32::WHITE };
                let g = p.layout_no_wrap(word.to_string(), FontId::proportional(11.0), col);
                let lr = Rect::from_min_size(q + vec2(14.0, -18.0), g.size()).expand(3.0);
                p.rect_filled(lr, 3.0, Color32::from_black_alpha(200));
                p.galley(lr.min + vec2(3.0, 3.0), g, col);
            }
            let icon = match (tool, under) {
                (Tool::Razor, _) if in_strip(q) && !in_ruler(q) => Some(egui::CursorIcon::Crosshair),
                _ if edge_mode.is_some() => edge_mode.map(|e| e.1),
                (Tool::Select, Hit::Clip { .. }) if self.press.is_some() => Some(egui::CursorIcon::Grabbing),
                _ => None,
            };
            if let Some(icon) = icon {
                ui.ctx().set_cursor_icon(icon);
            }
            // The razor shows where it would cut.
            if tool == Tool::Razor && in_strip(q) && !in_ruler(q) {
                let x = fx(px_to_frame(self.view, strip.min.x, q.x).round());
                let ys = match self.row_at(q.y) {
                    Some(ti) if !mods.shift => {
                        let i = ROWS.iter().position(|&t| t == ti).unwrap_or(0);
                        let rr = row_rect(i);
                        (rr.min.y, rr.max.y)
                    }
                    _ => (rect.min.y + RULER_H, rect.max.y),
                };
                p.line_segment([pos2(x, ys.0), pos2(x, ys.1)], Stroke::new(1.5, Color32::from_rgb(255, 120, 120)));
            }
        }

        // Something dragged here from the source viewer or the bin.
        let dragging_in = egui::DragAndDrop::has_payload_of_type::<Payload>(ui.ctx());
        if dragging_in {
            if let Some(q) = ui.input(|i| i.pointer.hover_pos()).filter(|q| strip.contains(*q) && !in_ruler(*q)) {
                let payload = egui::DragAndDrop::payload::<Payload>(ui.ctx());
                let f = px_to_frame(self.view, strip.min.x, q.x).round().max(0.0) as i64;
                let row = self.row_at(q.y);
                let col = Color32::from_rgb(255, 230, 120);
                match payload.as_deref() {
                    Some(Payload::Dissolve) => {
                        if let Some(at) = row.and_then(|t| seq.nearest_cut(t, f)) {
                            let i = ROWS.iter().position(|&t| Some(t) == row).unwrap_or(0);
                            let lane = row_rect(i);
                            let x = fx(at as f64);
                            p.rect_stroke(Rect::from_x_y_ranges((x - 10.0)..=(x + 10.0), lane.y_range()), 3.0, Stroke::new(2.0, col), egui::StrokeKind::Inside);
                        }
                    }
                    Some(pl) => {
                        let len = match pl {
                            Payload::Source { len } => *len,
                            Payload::Text => (crate::textui::TEXT_SECS * seq.fps.as_f64()).round() as i64,
                            _ => 0,
                        };
                        if let Some(i) = ROWS.iter().position(|&t| Some(t) == row) {
                            let lane = row_rect(i);
                            let x0 = fx(f as f64);
                            let x1 = if len > 0 { fx((f + len) as f64) } else { x0 + 40.0 };
                            p.rect_stroke(Rect::from_x_y_ranges(x0..=x1, (lane.min.y + 1.0)..=(lane.max.y - 1.0)), 2.0, Stroke::new(2.0, col), egui::StrokeKind::Inside);
                            let what = if mods.command { "insert" } else { "overwrite" };
                            p.text(pos2(x0 + 4.0, lane.min.y - 2.0), Align2::LEFT_BOTTOM, format!("{what} (cmd: insert)"), FontId::proportional(10.0), col);
                        }
                    }
                    None => {}
                }
                if ui.input(|i| i.pointer.any_released()) {
                    // The modifiers on the release count too (a test's event).
                    let rel = ui.input(|i| {
                        i.events.iter().fold(mods, |m, e| match e {
                            egui::Event::PointerButton { modifiers, pressed: false, .. } => m | *modifiers,
                            _ => m,
                        })
                    });
                    if let Some(pl) = egui::DragAndDrop::take_payload::<Payload>(ui.ctx()) {
                        ev = Event::Drop { payload: (*pl).clone(), frame: f, track: row, insert: rel.command };
                    }
                }
            }
        }

        // The right-click menu: what it opened on is kept while it is open.
        if r.secondary_clicked() && !self.navving {
            if let Some(q) = pointer.filter(|q| in_strip(*q) && !in_ruler(*q)) {
                let f = px_to_frame(self.view, strip.min.x, q.x).round().max(0.0) as i64;
                self.menu = Some((self.hit(seq, q), f, self.row_at(q.y)));
            }
        }
        if let Some((hit, f, row)) = self.menu {
            let mut pick = None;
            let open = r.context_menu(|ui| {
                pick = menu_ui(ui, seq, hit, f, row);
                if pick.is_some() {
                    ui.close();
                }
            });
            if let Some(c) = pick {
                ev = Event::Menu(c);
            }
            if open.is_none() {
                self.menu = None;
            }
        }

        if r.double_clicked() {
            if let Some(Hit::Clip { track, start }) = pointer.filter(|q| in_strip(*q) && !in_ruler(*q)).map(|q| self.hit(seq, q)) {
                ev = Event::Open { track, start };
            }
        }
        if r.drag_started_by(PointerButton::Primary) && !(mods.alt && !press_on_clip) {
            if let Some(q) = ui.input(|i| i.pointer.press_origin()).filter(|q| in_strip(*q)) {
                let hit = if in_ruler(q) { Hit::Empty } else { self.hit(seq, q) };
                // The razor never drags clips; Rate only drags edges.
                let hit = match (tool, hit) {
                    (_, h @ (Hit::Clip { .. } | Hit::Edge { .. })) if hold.is_some() => h,
                    (Tool::Razor, _) | (Tool::Rate | Tool::Roll, Hit::Clip { .. }) => Hit::Empty,
                    (_, h) => h,
                };
                self.press = Some((hit, px_to_frame(self.view, strip.min.x, q.x), q.y));
                if hit == Hit::Empty && !in_ruler(q) && tool == Tool::Select {
                    self.marquee = Some(q);
                }
            }
        }
        let frame_at = |x: f32| px_to_frame(self.view, strip.min.x, x);
        match self.press {
            Some((hit, f0, y0)) if hit != Hit::Empty => {
                if let Some(q) = pointer {
                    let df = frame_at(q.x) - f0;
                    let dtrack = -((q.y - y0) / (track_h + GAP)).round() as i32;
                    let done = r.drag_stopped();
                    ev = Event::Drag { hit, df, dtrack, mods, done };
                    if done {
                        self.press = None;
                        self.press_mods = egui::Modifiers::NONE;
                    }
                }
                if !r.dragged() && !r.drag_stopped() {
                    self.press = None;
                }
            }
            _ if self.marquee.is_some() => {
                let a = self.marquee.unwrap_or_default();
                let done = r.drag_stopped() || !ui.input(|i| i.pointer.primary_down());
                if let Some(q) = pointer {
                    let mr = Rect::from_two_pos(a, q);
                    p.rect_filled(mr, 0.0, v.selection.bg_fill.linear_multiply(0.15));
                    p.rect_stroke(mr, 0.0, Stroke::new(1.0, v.selection.stroke.color), egui::StrokeKind::Inside);
                    let (f0, f1) = (frame_at(mr.min.x), frame_at(mr.max.x));
                    let mut picks = Vec::new();
                    for (i, &ti) in ROWS.iter().enumerate() {
                        if !row_rect(i).intersects(mr) {
                            continue;
                        }
                        for c in &seq.tracks[ti].clips {
                            if (c.start as f64) < f1 && (c.end() as f64) > f0 {
                                picks.push((ti, c.start));
                            }
                        }
                    }
                    ev = Event::Marquee { picks, add: mods.shift, done };
                }
                if done {
                    self.marquee = None;
                    self.press = None;
                    self.press_mods = egui::Modifiers::NONE;
                }
            }
            _ => {
                if r.drag_stopped() {
                    self.press = None;
                }
                if !ui.input(|i| i.pointer.any_down()) && self.press.is_none() && !clicked {
                    self.press_mods = egui::Modifiers::NONE;
                }
                let scrub = r.dragged_by(PointerButton::Primary) && self.press.is_some();
                if clicked {
                    if let Some(q) = pointer.filter(|q| in_strip(*q)) {
                        let frame = frame_at(q.x);
                        ev = if in_ruler(q) {
                            Event::Scrub(frame.floor().max(0.0) as i64)
                        } else {
                            let hit = self.hit(seq, q);
                            // An empty lane (or the razor anywhere on a track)
                            // still says which track it was on.
                            let hit = match (hit, self.row_at(q.y)) {
                                (Hit::Empty, Some(t)) if tool == Tool::Razor => seq.tracks[t]
                                    .at(frame.round() as i64)
                                    .map_or(Hit::Empty, |c| Hit::Clip { track: t, start: c.start }),
                                (h, _) => h,
                            };
                            Event::Click { hit, add: mods.shift || mods.command, frame, mods }
                        };
                    }
                } else if scrub {
                    if let Some(q) = pointer {
                        ev = Event::Scrub(frame_at(q.x).floor().max(0.0) as i64);
                    }
                }
            }
        }

        // Marks and playhead on top.
        for (m, inward) in [(marks.inp, 1.0), (marks.out.map(|o| o + 1), -1.0)] {
            if let Some(m) = m {
                let x = fx(m as f64);
                let c = v.selection.stroke.color;
                p.line_segment([pos2(x, rect.min.y), pos2(x, rect.max.y)], Stroke::new(1.0, c));
                p.line_segment([pos2(x, rect.min.y), pos2(x + 6.0 * inward, rect.min.y)], Stroke::new(2.0, c));
            }
        }
        let x = fx(playhead as f64);
        let x2 = fx(playhead as f64 + 1.0);
        if x2 - x > 3.0 {
            p.rect_filled(Rect::from_x_y_ranges(x..=x2, ruler.y_range()), 0.0, v.selection.stroke.color.gamma_multiply(0.4));
        }
        p.line_segment([pos2(x, rect.min.y), pos2(x, rect.max.y)], Stroke::new(1.5, Color32::from_rgb(220, 60, 60)));
        ev
    }
}

/// The right-click menu for what is under the pointer.
fn menu_ui(ui: &mut Ui, seq: &Sequence, hit: Hit, f: i64, row: Option<usize>) -> Option<Cmd> {
    let mut pick = None;
    // The cut this is about: the edge or dissolve hit, else the nearest
    // cut on the row.
    let cut = match hit {
        Hit::Edge { track, at, .. } | Hit::Dissolve { track, at, .. } => Some((track, at)),
        _ => row.and_then(|t| seq.nearest_cut(t, f).map(|at| (t, at))),
    };
    if let Some((track, at)) = cut {
        let name = seq.tracks[track].name;
        let tc = crate::editor::timecode(at.max(0) as u64, seq.fps.as_f64());
        ui.weak(format!("{name} cut at {tc}"));
        let (a, b) = seq.tracks[track].cut(at);
        let both = a.is_some() && b.is_some();
        if seq.tracks[track].dissolve_at(at).is_some() {
            if ui.button("remove dissolve  (cmd shift D)").clicked() {
                pick = Some(Cmd::RemoveDissolve { track, at });
            }
        } else if ui.add_enabled(both, egui::Button::new("add dissolve  (cmd D)")).on_disabled_hover_text("a dissolve needs a clip on both sides of the cut").clicked() {
            pick = Some(Cmd::AddDissolve { track, at });
        }
        if ui.button("select edit point  (G)").clicked() {
            pick = Some(Cmd::SelectEdit { track, at });
        }
        ui.separator();
    }
    if let Some(track) = row {
        if let Some(c) = seq.tracks[track].at(f) {
            let start = c.start;
            ui.weak(seq.media[c.media].name.clone());
            if ui.button("razor here  (X tool, click)").clicked() {
                pick = Some(Cmd::Razor { track, at: f, all: false });
            }
            if ui.button("razor every track here").clicked() {
                pick = Some(Cmd::Razor { track, at: f, all: true });
            }
            if ui.button("speed…  (cmd R)").clicked() {
                pick = Some(Cmd::Speed { track, start });
            }
            if !seq.media[c.media].generated {
                let m = c.media;
                let path = seq.media[m].path.clone();
                ui.menu_button(format!("input colour: {}", seq.colour.input_of(&path).label()), |ui| {
                    if let Some(sp) = crate::editor::input_space_menu(ui, &seq.colour, &path) {
                        pick = Some(Cmd::Input { media: m, space: sp });
                    }
                });
            }
            if ui.button("delete, leave a gap  (del)").clicked() {
                pick = Some(Cmd::Lift { track, start });
            }
            if ui.button("ripple delete  (shift del)").clicked() {
                pick = Some(Cmd::RippleDelete { track, start });
            }
        }
    }
    if pick.is_none() && cut.is_none() && row.and_then(|t| seq.tracks[t].at(f)).is_none() {
        ui.weak("nothing here");
    }
    pick
}
