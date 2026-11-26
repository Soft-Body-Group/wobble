//! The window: the bin on the left, the source and record viewers side by
//! side, the timeline under them. Hotkeys do the editing (`KEYS`).
//!
//! Focus is one of bin / source / record (click, or shift+1/2/3). The
//! transport keys drive the last-focused VIEWER even while the bin has
//! focus; I / O mark the focused viewer's side; comma inserts and period
//! overwrites from the source marks into the record marks (three-point
//! rules in `model::resolve`) on the targeted tracks.

use crate::bin::{self, Bin};
use crate::program::{Pick, Program};
use crate::timelineui::{self, TimelineView};
use crate::transport::Transport;
use crate::viewer::ViewerPaint;
use egui::{pos2, vec2, Align2, Color32, FontId, Key, Modifiers, Rect, Sense, Stroke};
use imfram::history::History;
use imfram::hudkeys::{self, HudKey};
use imfram::timeline::{self, Timeline};
use media::{Cache, FrameServer, PlayState, SourceInfo};
use model::{Fps, Marks, Sequence, Targets};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// What the prefetch assumes the display refreshes at.
const DISPLAY_HZ: f64 = 60.0;
const HUD_FONT: f32 = 11.0;
const CACHE_BAR_H: f32 = 3.0;
const MENU_H: f32 = 22.0;
/// The height of one row of key hints.
const HINT_ROW: f32 = 24.0;
/// Points of horizontal drag per frame when scrubbing a viewer's picture.
pub const SCRUB_PX: f32 = 5.0;
/// The clear-in/out buttons' fill: red, to be seen.
pub const CLEAR_MARKS: Color32 = Color32::from_rgb(176, 42, 42);
/// Between two rows of key hints.
const HINT_GAP: f32 = 10.0;
const TITLE_H: f32 = 20.0;
const SCRUB_H: f32 = 58.0;

pub const KEYS: &[HudKey] = &[
    hudkeys::key("J K L", "shuttle (again: faster)", Key::K),
    hudkeys::key("space", "play / stop", Key::Space),
    hudkeys::key("left right", "frame step", Key::ArrowLeft),
    hudkeys::key("I O", "mark in / out", Key::I),
    hudkeys::chord("shift I O", "go to in / out"),
    hudkeys::chord("alt I O X", "clear in / out / both"),
    hudkeys::key(",", "insert", Key::Comma),
    hudkeys::key(".", "overwrite", Key::Period),
    hudkeys::key("M", "append at the end", Key::M),
    hudkeys::key("up down", "previous / next edit", Key::ArrowUp),
    hudkeys::chord("1-3  4-7", "target V1-V3 / A1 A2 A3 M"),
    hudkeys::chord("shift 1 2 3", "focus bin / source / record"),
    hudkeys::chord("= - \\", "zoom timeline / fit"),
    hudkeys::chord("cmd Z", "undo (shift: redo)"),
    hudkeys::chord("cmd T", "optimise source"),
    hudkeys::key("D", "tap: select clip (shift: next track)", Key::D),
    hudkeys::chord("hold A D + drag", "trim in / out (shift: stretch)"),
    hudkeys::chord("hold Z C + drag", "ripple in / out"),
    hudkeys::chord("hold S + drag", "slip (shift: speed at playhead)"),
    hudkeys::chord("del", "lift clip (shift: ripple delete)"),
    hudkeys::key("G", "select edit (again: out / in / roll)", Key::G),
    hudkeys::chord("alt left right", "trim edit (shift: 10)"),
    hudkeys::key("B", "ripple / plain trims", Key::B),
    hudkeys::chord("V X N Y U", "tools: select razor roll slip slide"),
    hudkeys::chord("cmd K", "cut at playhead (shift: all tracks)"),
    hudkeys::key("E", "extend edit to playhead", Key::E),
    hudkeys::chord("alt , .", "slip clip (shift: 10)"),
    hudkeys::chord("cmd R", "clip speed"),
    hudkeys::chord("shift .", "fit to fill"),
    hudkeys::chord("cmd D", "dissolve on cut / 2 clips (alt = -)"),
    hudkeys::key("F", "reframe: fit / fill (alt: reset)", Key::F),
    hudkeys::chord("[ ]", "reframe: zoom (shift: fine)"),
    hudkeys::chord("cmd arrows", "reframe: move (shift: 10%)"),
    hudkeys::chord("cmd M", "render"),
    hudkeys::chord("cmd shift T", "new text (Enter edits)"),
    hudkeys::chord("drag", "move (cmd insert, alt copy)"),
    hudkeys::chord("drag edge", "trim (cmd: ripple)"),
    hudkeys::chord("drag lane", "marquee select"),
    hudkeys::chord("drag picture", "scrub (shift: source clip to timeline)"),
    hudkeys::chord("right-click", "cut menu: dissolve, razor"),
    hudkeys::key("T", "snapping", Key::T),
    hudkeys::chord("R  shift R", "over a viewer: fit / match both"),
    hudkeys::chord("H  shift H", "key hints / readouts"),
];

/// A viewer's key hints in motion (`Editor::hints_over`): how far they are
/// shown (0 hidden .. 1) and each row's words out (0..1).
#[derive(Clone, Default)]
struct HintAnim {
    vis: f32,
    rows: Vec<f32>,
}

/// `v` a step `k` of the way to `target`, settling on it when close. True
/// while it is still on its way (keep repainting).
fn approach(v: &mut f32, target: f32, k: f32) -> bool {
    *v += (target - *v) * k;
    if (target - *v).abs() > 0.004 {
        true
    } else {
        *v = target;
        false
    }
}

/// Ease out: quick to start, gentle to settle.
fn ease(t: f32) -> f32 {
    1.0 - (1.0 - t) * (1.0 - t)
}

pub fn style(ctx: &egui::Context) {
    // Wobble's dark (imfram's `DARK` overrides: near-black background, dim
    // borders, teal-grey buttons) rather than the core table's lighter
    // defaults, so the two apps look like one. Overrides, so every linked
    // slot (windows, fields, canvases) follows.
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let dark = imfram::theme::DARK.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        imfram::theme::install(&dark);
    });
    imfram::theme::sync(ctx);
    ctx.all_styles_mut(|s| s.animation_time /= 3.0);
}

#[derive(Debug)]
pub struct Summary {
    pub duration: i64,
    pub playhead: i64,
    pub clips: Vec<(usize, i64, i64, i64)>,
    /// (track, cut, before, after)
    pub dissolves: Vec<(usize, i64, i64, i64)>,
    pub speeds: Vec<(usize, i64, f64)>,
}

/// The window's two modes: EDIT (bin, two-up, timeline) and SELECTS
/// (one viewer, the clips listed with their verdicts, the selects bin of
/// subclips under it). ` (backtick) or the menubar switches.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Mode {
    #[default]
    Edit,
    Selects,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Focus {
    Bin,
    Source,
    Record,
}

pub(crate) struct SourceClip {
    pub(crate) server: FrameServer,
    pub(crate) transport: Transport,
    pub(crate) scrub: Timeline,
    pub(crate) marks: Marks,
    pub(crate) info: SourceInfo,
    /// The file the user opened (the server may read its optimised copy).
    pub(crate) original: PathBuf,
    pub(crate) optimised: bool,
    pub(crate) shown: Option<u64>,
    /// Marks put on by a drop, an append or a subclip load: while they are
    /// still these, leaving the clip does not remember them as its own.
    pub(crate) set_marks: Option<Marks>,
}

pub(crate) struct Record {
    pub(crate) seq: Sequence,
    pub(crate) history: History<Sequence>,
    pub(crate) transport: Transport,
    pub(crate) marks: Marks,
    pub(crate) program: Program,
    pub(crate) view: TimelineView,
    pub(crate) targets: Targets,
    pub(crate) shown: Option<i64>,
    /// Picture size last shown (a gap keeps the frame's shape).
    pub(crate) dims: (u32, u32),
    /// The selected clip or edit point.
    pub(crate) sel: crate::timelineui::Selection,
    /// The reframe of the picture on screen (kept while a frame decodes).
    pub(crate) shown_reframe: model::Reframe,
    /// The input transform of the picture on screen (kept with it).
    pub(crate) shown_idt: Option<Arc<crate::viewer::Idt>>,
    /// Dragged edges snap (T).
    pub(crate) snap: bool,
    /// A frame to show instead of the playhead's (a dragged clip's head).
    pub(crate) peek: Option<i64>,
    /// Trims ripple (move what follows) or not.
    pub(crate) ripple: bool,
    /// What a press on the timeline does (V C R Y U).
    pub(crate) tool: crate::timelineui::Tool,
}

/// How far past the last clip the record playhead may go (click, drag,
/// step): edits there leave a gap before them.
const PAST_END_SECS: f64 = 600.0;

impl Record {
    /// The record transport's extent after the sequence changed: the
    /// playhead may run `PAST_END_SECS` past the end (or past wherever it
    /// already is), and playback still stops at the end.
    pub(crate) fn fit_transport(&mut self) {
        let d = self.seq.duration().max(0) as u64;
        let past = (PAST_END_SECS * self.seq.fps.as_f64()) as u64;
        self.transport.end = Some(d);
        self.transport.frames = (d + 1).max(self.transport.frame() + 1) + past;
    }
}

/// Per-viewer counters for the HUD.
#[derive(Default)]
pub(crate) struct Meter {
    times: Vec<Instant>,
    total: u64,
}

impl Meter {
    fn shown(&mut self) {
        self.times.push(Instant::now());
        self.total += 1;
    }

    fn rate(&mut self) -> usize {
        let now = Instant::now();
        self.times.retain(|&t| now.duration_since(t).as_secs_f32() < 1.0);
        self.times.len()
    }
}

pub struct Editor {
    /// EXPERIMENT: wobble hosted in process (fold.rs).
    #[cfg(feature = "fold")]
    pub fold: crate::fold::Fold,
    pub(crate) cache: Arc<Cache>,
    pub opt: crate::optimiseui::OptimiseUi,
    pub(crate) source: Option<SourceClip>,
    pub(crate) rec: Record,
    pub(crate) bin: Bin,
    pub(crate) focus: Focus,
    /// The viewer the transport keys drive.
    pub(crate) monitor: Focus,
    pub(crate) error: Option<String>,
    /// A short message for the record info strip, and when it was set.
    pub(crate) notice: Option<(String, Instant)>,
    /// The speed dialog: open, and its value.
    pub(crate) speed_dialog: Option<f64>,
    pub(crate) speed_dialog_focus: bool,
    pub(crate) project: crate::project::ProjectState,
    pub(crate) export: crate::exportui::ExportUi,
    pub(crate) panes: crate::panes::Panes,
    /// The panels' dock (`panes`): the main window first, then torn-off
    /// windows. Out of its slot while a window draws through it.
    pub(crate) docks: Option<crate::panes::Docks>,
    /// A saved layout to restore once windows can be made (`about_to_wait`).
    pub(crate) dock_restore: Option<crate::panes::Saved>,
    /// window > reset layout, taken up between frames.
    pub(crate) dock_reset: bool,
    /// The playhead as the sound last set it (a different one is a seek).
    pub(crate) audio_set: Option<f64>,
    /// The panel wearing the key-target edge last frame (tests).
    pub(crate) key_panel: Option<crate::panes::Pane>,
    /// A viewer's picture being drag-scrubbed: (frame at the press, x).
    pub(crate) view_scrub: [Option<(i64, f32)>; 2],
    /// The modifiers on the last pointer press (with the live ones, what a
    /// drag on a picture means).
    pub(crate) press_mods: Modifiers,
    /// The screen recorder and the screenshot (wobble's), shared with the
    /// app's draw loop, which copies the presented frame for them.
    pub(crate) recorder: Arc<std::sync::Mutex<crate::recorder::Recorder>>,
    pub(crate) shot: Arc<std::sync::Mutex<crate::recorder::shot::Shot>>,
    /// The window's device (the recorder's readbacks).
    pub(crate) device: Option<egui_wgpu::wgpu::Device>,
    /// Where each viewer's picture was drawn last (tests read pixels there).
    pub(crate) pic_rects: std::cell::Cell<[Option<Rect>; 2]>,
    /// Input transforms by (input space, working space), built once.
    pub(crate) idts: std::cell::RefCell<std::collections::HashMap<String, Arc<crate::viewer::Idt>>>,
    /// "cache shot locally" runs: (name, frames done, frames, stop).
    pub(crate) local_caching: Vec<(String, Arc<std::sync::atomic::AtomicU64>, u64, Arc<std::sync::atomic::AtomicBool>)>,
    /// Panels closed with W, and their window's layout then (put back on
    /// reopening when nothing else moved meanwhile, as wobble does).
    pub(crate) closed: Vec<(crate::panes::Pane, usize, imfram::dock::Tree<crate::panes::Pane>)>,
    /// A timeline drag in progress, and what it draws.
    pub(crate) drag: Option<crate::mouse::Drag>,
    pub(crate) drag_cancelled: bool,
    pub(crate) overlay: crate::timelineui::Overlay,
    /// The organisicator marketing post this project belongs to.
    pub(crate) post: Option<crate::marketing::PostCtx>,
    pub(crate) posts_window: Option<crate::marketing::PostsWindow>,
    /// The relink window is open.
    pub(crate) relink_open: bool,
    /// The sequence settings window: open, and its fields.
    pub(crate) seq_settings: Option<(u32, u32, i64, i64)>,
    /// The sound output, opened on first use; `Err` says why there is none.
    pub(crate) audio: Option<Result<crate::audio::Audio, String>>,
    /// After a stall, sound stays off until then.
    pub(crate) audio_off_until: Option<Instant>,
    pub(crate) last_tick: Instant,
    pub(crate) meters: [Meter; 2],
    pub upload_ms: [f32; 2],
    pub quit: bool,
    /// The scripting API's listener (`apiserve`), when it is open.
    pub(crate) api: Option<api::server::Host>,
    /// A scripting call is running: edits leave the history to it.
    pub(crate) api_quiet: bool,
    /// The source (0) and record (1) viewers' zoom and pan.
    pub(crate) views: [crate::viewnav::View; 2],
    /// Titles: the viewer's text, the text window, the style new text takes.
    pub(crate) textv: crate::textui::TextView,
    /// The clip browser (seq_manip's), when built in.
    #[cfg(feature = "clips")]
    pub(crate) clips: Option<crate::clipbrowser::ClipBrowser>,
    /// The clip last slipped (track, start, when): its handles band and the
    /// first / last frame two-up show while it is recent or dragging.
    pub(crate) slip_view: Option<(usize, i64, Instant)>,
    /// Wobble clips (experimental).
    #[cfg(feature = "wobble")]
    pub(crate) wob: crate::wobbleclip::Wob,
    /// The hold key held now (A S D Z C, shift A S D), and whether a drag
    /// used it (then its tap action does not fire).
    pub(crate) hold: Option<crate::mouse::HoldOp>,
    pub(crate) hold_used: bool,
    /// A plain D went down (shift or not): its tap fires on release.
    pub(crate) d_tap: Option<bool>,
    /// Shift was down when the hold key went down; or cmd / alt (a chord,
    /// not a hold).
    pub(crate) hold_shift: bool,
    pub(crate) hold_chord: bool,
    /// The first-launch hotkey tour (help > hotkey tour replays it).
    pub(crate) tour: crate::tour::Tour,
    /// "request a feature" (the menubar's red button).
    pub(crate) feedback: crate::feedback::Feedback,
    /// Edit or selects (`selects.rs`).
    pub(crate) mode: Mode,
    /// Which clips selects mode lists.
    pub(crate) filter: crate::bin::Filter,
}

impl Editor {
    pub fn new(headless: bool) -> Self {
        let cache = Cache::new(Cache::default_budget());
        let seq = Sequence::new(Fps::new(24, 1));
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_else(|| ".".into());
        Self {
            opt: crate::optimiseui::OptimiseUi::new(),
            source: None,
            rec: Record {
                history: History::new(&seq),
                transport: {
                    let mut t = Transport::new(1, 24.0);
                    // The record side stops at the end, like every NLE.
                    t.looping = false;
                    t
                },
                marks: Marks::default(),
                program: Program::new(cache.clone()),
                view: TimelineView::new(),
                targets: Targets::default(),
                shown: None,
                dims: (1920, 1080),
                sel: crate::timelineui::Selection::None,
                shown_reframe: model::Reframe::default(),
                shown_idt: None,
                snap: true,
                peek: None,
                // Plain trims by default (B toggles ripple), as in Premiere.
                ripple: false,
                tool: Default::default(),
                seq,
            },
            cache,
            bin: Bin::new(home),
            focus: Focus::Source,
            monitor: Focus::Source,
            error: None,
            notice: None,
            speed_dialog: None,
            speed_dialog_focus: false,
            #[cfg(feature = "fold")]
            fold: Default::default(),
            project: crate::project::ProjectState::new(headless),
            export: crate::exportui::ExportUi::default(),
            panes: crate::panes::Panes::load(headless),
            docks: None,
            dock_restore: None,
            dock_reset: false,
            view_scrub: [None, None],
            key_panel: None,
            audio_set: None,
            press_mods: Modifiers::NONE,
            recorder: Arc::new(std::sync::Mutex::new(crate::recorder::Recorder::new(crate::panes::Panes::load(headless).rec_opts.unwrap_or_default()))),
            shot: Default::default(),
            device: None,
            pic_rects: Default::default(),
            idts: Default::default(),
            local_caching: Vec::new(),
            closed: Vec::new(),
            drag: None,
            drag_cancelled: false,
            overlay: Default::default(),
            post: None,
            posts_window: None,
            relink_open: true,
            seq_settings: None,
            audio: None,
            audio_off_until: None,
            last_tick: Instant::now(),
            meters: [Meter::default(), Meter::default()],
            upload_ms: [0.0; 2],
            quit: false,
            api: None,
            api_quiet: false,
            views: Default::default(),
            textv: Default::default(),
            #[cfg(feature = "clips")]
            clips: Some(crate::clipbrowser::ClipBrowser::new(headless)),
            slip_view: None,
            #[cfg(feature = "wobble")]
            wob: Default::default(),
            hold: None,
            hold_used: false,
            d_tap: None,
            hold_shift: false,
            hold_chord: false,
            tour: crate::tour::Tour::new(crate::panes::Panes::load(headless).tour_seen, headless),
            feedback: crate::feedback::Feedback::default(),
            mode: Mode::Edit,
            filter: Default::default(),
        }
    }

    // --- opening -----------------------------------------------------------

    /// Load a file into the source viewer (through its optimised copy when
    /// one matches), and into the bin.
    pub fn open(&mut self, path: &Path) {
        let optimised = media::optimise::find(path, &self.opt.settings);
        let read = optimised.as_deref().unwrap_or(path);
        match media::open(read) {
            Ok(source) => {
                let info = source.info().clone();
                let server = FrameServer::new(source, self.cache.clone());
                let mut scrub = Timeline::new(info.first, info.first + info.frames as i64 - 1);
                // The transport owns the clock; the timeline only scrubs.
                scrub.fps = 0.0;
                let detail = format!("{}x{} {} fr", info.width, info.height, info.frames);
                // The clip leaving keeps its marks (put back next time).
                self.remember_source_marks();
                self.bin.add(path, detail);
                let marks = self.bin.clip(path).map(|i| self.bin.items[i].marks).unwrap_or_default();
                if let Some(d) = path.parent() {
                    if self.bin.items.len() == 1 {
                        self.bin.go(d.to_path_buf());
                    }
                }
                self.source = Some(SourceClip {
                    server,
                    transport: Transport::new(info.frames, info.fps()),
                    scrub,
                    marks,
                    info,
                    original: path.to_path_buf(),
                    optimised: optimised.is_some(),
                    shown: None,
                    set_marks: None,
                });
                self.error = None;
                self.focus = Focus::Source;
                self.monitor = Focus::Source;
            }
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    /// Re-open the source clip (an optimise landed, or a purge), keeping
    /// its playhead, marks and speed; the record side re-opens its files.
    pub fn reopen(&mut self) {
        self.rec.program.reset();
        let Some(c) = &self.source else { return };
        let (path, pos, speed, marks) = (c.original.clone(), c.transport.pos, c.transport.speed, c.marks);
        let focus = (self.focus, self.monitor);
        self.open(&path);
        (self.focus, self.monitor) = focus;
        if let Some(c) = &mut self.source {
            c.transport.pos = pos.min(c.transport.frames as f64 - 1.0);
            c.transport.speed = speed;
            c.marks = marks;
        }
    }

    /// The source clip: its path, read through an optimised copy, intra.
    pub fn current(&self) -> Option<(&Path, bool, bool)> {
        self.source.as_ref().map(|c| (c.original.as_path(), c.optimised, c.info.intra))
    }

    // --- headless hooks ------------------------------------------------------

    /// Put the active viewer's playhead on a displayed frame number.
    pub fn seek_display(&mut self, f: i64) {
        match self.monitor {
            Focus::Record => self.rec.transport.seek(f.max(0) as u64),
            _ => {
                if let Some(c) = &mut self.source {
                    c.transport.seek((f - c.info.first).max(0) as u64);
                }
            }
        }
    }

    pub fn focus_record(&mut self) {
        self.focus = Focus::Record;
        self.monitor = Focus::Record;
    }

    /// Start the active viewer playing at a shuttle speed (`--play`).
    pub fn play(&mut self, speed: f64) {
        if let Some(t) = self.active_transport() {
            t.set_speed(speed);
        }
        self.last_tick = Instant::now();
    }

    fn active_transport(&mut self) -> Option<&mut Transport> {
        match self.monitor {
            Focus::Record => Some(&mut self.rec.transport),
            _ => self.source.as_mut().map(|c| &mut c.transport),
        }
    }

    /// Stop the sound before the app goes.
    pub fn shutdown(&mut self) {
        self.project_clean_exit();
        if let Some(Ok(a)) = &mut self.audio {
            a.stop();
        }
        self.audio = None;
    }

    /// Sound, in words: playing or not, the clock, what is buffered.
    pub fn audio_status(&self) -> String {
        match &self.audio {
            None => "audio: not opened".into(),
            Some(Err(e)) => format!("audio: {e}"),
            Some(Ok(a)) => {
                let (ring, secs) = a.status();
                format!("audio: {} Hz, {}, clock {secs:.2}s, {ring} frames ahead", a.rate, if a.is_playing() { "playing" } else { "idle" })
            }
        }
    }

    pub fn dropped(&self) -> u64 {
        match self.monitor {
            Focus::Record => self.rec.transport.dropped,
            _ => self.source.as_ref().map_or(0, |c| c.transport.dropped),
        }
    }

    /// Distinct pictures the active viewer has put on screen.
    pub fn shown_total(&self) -> u64 {
        self.meters[(self.monitor == Focus::Record) as usize].total
    }

    pub fn frame_display(&self) -> Option<i64> {
        match self.monitor {
            Focus::Record => Some(self.rec.transport.frame() as i64),
            _ => self.source.as_ref().map(|c| c.info.first + c.transport.frame() as i64),
        }
    }

    /// Block until the active viewer's frame is decoded (`--shot`).
    pub fn wait_for_picture(&mut self, timeout: std::time::Duration) {
        self.feed();
        match self.monitor {
            Focus::Record => self.rec.program.wait_for(&self.rec.seq, self.rec.transport.frame() as i64, timeout),
            _ => {
                if let Some(c) = &self.source {
                    c.server.wait_for(c.transport.frame(), timeout);
                }
            }
        }
    }

    /// `--cuts N`: cut the source into N short clips, each from somewhere
    /// else in it, overwritten end to end (alternating V1 / V2): a
    /// sequence with a cut every half second, for measuring playback
    /// across cuts.
    pub fn build_cuts(&mut self, n: usize, fx: bool) {
        let Some(mut c) = self.source.as_mut() else { return };
        let frames = c.info.frames as i64;
        let len = 12.min(frames);
        for k in 0..n as i64 {
            let src_in = (k * 37 + 5) % (frames - len).max(1);
            c.marks = Marks { inp: Some(src_in), out: Some(src_in + len - 1) };
            self.rec.marks = Marks::default();
            self.rec.fit_transport();
            self.rec.transport.seek(self.rec.seq.duration() as u64);
            self.rec.targets.set_video(Some(if fx { model::V1 } else { [model::V1, model::V2][k as usize % 2] }));
            self.edit(false);
            let Some(next) = &mut self.source else { return };
            c = next;
        }
        if fx {
            // A speed change on every third clip (2x, 0.5x, backwards) and
            // an 8-frame dissolve on every other cut, all on V1.
            let speeds = [2.0, 0.5, -1.0];
            let starts: Vec<i64> = self.rec.seq.tracks[model::V1].clips.iter().map(|c| c.start).collect();
            for (k, st) in starts.iter().enumerate() {
                if k % 3 == 1 {
                    if let Some(i) = self.rec.seq.tracks[model::V1].clips.iter().position(|c| c.start == *st) {
                        self.rec.seq.set_speed(model::V1, i, speeds[(k / 3) % 3]);
                    }
                }
            }
            let cuts: Vec<i64> = self.rec.seq.tracks[model::V1].clips.iter().skip(1).map(|c| c.start).collect();
            for (k, at) in cuts.iter().enumerate() {
                if k % 2 == 0 {
                    let _ = self.rec.seq.add_dissolve(model::V1, *at, 8);
                }
            }
            self.changed(false);
        }
        self.rec.targets = Targets::default();
        self.rec.transport.seek(0);
        self.rec.view.fit();
    }

    /// The sequence in plain numbers, for the key test: duration, record
    /// playhead, and every clip as (track, start, len, src_in).
    pub fn sequence_summary(&self) -> Summary {
        let r = &self.rec;
        let clips = r.seq.tracks.iter().enumerate().flat_map(|(ti, t)| t.clips.iter().map(move |c| (ti, c.start, c.len, c.src_in))).collect();
        let dissolves = r.seq.tracks.iter().enumerate().flat_map(|(ti, t)| t.dissolves.iter().map(move |d| (ti, d.at, d.before, d.after))).collect();
        let speeds = r.seq.tracks.iter().enumerate().flat_map(|(ti, t)| t.clips.iter().map(move |c| (ti, c.start, c.speed))).collect();
        Summary { duration: r.seq.duration(), playhead: r.transport.frame() as i64, clips, dissolves, speeds }
    }

    /// cmd+I / file > import / the bin's browse button: the clip browser
    /// (seq_manip's); in a build without it, a file dialog into the bin.
    pub(crate) fn import(&mut self) {
        #[cfg(feature = "clips")]
        {
            self.open_clips();
        }
        #[cfg(not(feature = "clips"))]
        {
            let mut d = imfram::filedialog::FileDialog::new();
            if self.bin.dir.is_dir() {
                d = d.set_directory(&self.bin.dir);
            }
            if let Some(files) = d.pick_files() {
                for f in &files {
                    self.bin.add(f, String::new());
                }
                if let Some(f) = files.first() {
                    self.open(f);
                }
            }
        }
    }

    /// The last HUD notice (tests).
    pub fn last_notice(&self) -> String {
        self.notice.as_ref().map(|n| n.0.clone()).unwrap_or_default()
    }

    /// How many clips are selected (tests).
    pub fn selected_clips(&self) -> usize {
        match &self.rec.sel {
            crate::timelineui::Selection::Clip { .. } => 1,
            crate::timelineui::Selection::Clips(v) => v.len(),
            _ => 0,
        }
    }

    pub fn targets(&self) -> Targets {
        self.rec.targets
    }

    pub fn pane_rect(&self, p: crate::panes::Pane) -> Option<Rect> {
        self.panes.rect_of(p)
    }

    pub fn readouts(&self) -> bool {
        self.panes.readouts
    }

    pub fn viewer_zoom(&self, i: usize) -> (f32, egui::Vec2) {
        (self.views[i].zoom, self.views[i].pan)
    }

    pub fn monitor_split(&self) -> f32 {
        match (self.pane_rect(crate::panes::Pane::Source), self.pane_rect(crate::panes::Pane::Record)) {
            (Some(a), Some(b)) => a.width() / (a.width() + b.width()).max(1.0),
            _ => 0.5,
        }
    }

    pub fn record_marks(&self) -> Marks {
        self.rec.marks
    }

    /// `--subs N`: N subtitles on V3 across the sequence, a different line
    /// each, in the subtitle style (playback and render checks).
    pub fn build_subs(&mut self, n: usize) {
        let d = self.rec.seq.duration().max(48);
        let len = (d / n.max(1) as i64).max(1);
        self.rec.seq.ensure_styles();
        let st = self.rec.seq.style_named("subtitle").unwrap_or(0);
        for k in 0..n as i64 {
            self.rec.seq.add_text(model::V3, k * len, len, &format!("Subtitle number {}\nsecond line {}", k + 1, k * 7 % 13), st);
        }
        self.changed(false);
    }

    /// The text window's clip, if open (tests).
    pub fn text_window_on(&self) -> Option<(usize, i64)> {
        self.textv.open
    }

    /// A clip's words and style name (tests).
    pub fn clip_text(&self, track: usize, start: i64) -> Option<(String, String)> {
        let c = self.rec.seq.tracks[track].clips.iter().find(|c| c.start == start)?;
        let t = c.text.as_ref()?;
        Some((t.text.clone(), self.rec.seq.styles.get(t.style)?.name.clone()))
    }

    pub fn style_pos(&self, name: &str) -> Option<(f32, f32)> {
        self.rec.seq.styles.iter().find(|s| s.name == name).map(|s| (s.x, s.y))
    }

    /// Where texts were drawn in the record viewer last frame (tests).
    pub fn text_boxes(&self) -> Vec<(usize, i64, Rect)> {
        self.textv.boxes.iter().map(|b| (b.0, b.1, b.3)).collect()
    }

    /// The slip band and whether the viewers are the slip two-up (tests).
    pub fn slip_state(&self) -> (Option<(usize, i64, i64, i64, i64)>, bool) {
        (self.slip_band(), self.slipping().is_some() && self.trim_frames().is_some())
    }

    #[cfg_attr(not(feature = "clips"), allow(dead_code))]
    pub fn source_path(&self) -> Option<PathBuf> {
        self.source.as_ref().map(|c| c.original.clone())
    }

    pub fn select_clip_for_test(&mut self, track: usize, start: i64) {
        self.rec.sel = crate::timelineui::Selection::Clip { track, start };
    }

    pub fn timeline_cmd_select_none(&mut self) {
        self.rec.sel = crate::timelineui::Selection::None;
    }

    pub fn set_tool_for_test(&mut self, t: crate::timelineui::Tool) {
        self.rec.tool = t;
    }

    pub fn snapping(&self) -> bool {
        self.rec.snap
    }

    pub fn ripple_trims(&self) -> bool {
        self.rec.ripple
    }

    pub fn tool(&self) -> crate::timelineui::Tool {
        self.rec.tool
    }

    pub fn set_record_marks(&mut self, m: Marks) {
        self.rec.marks = m;
    }

    pub fn audible_tracks(&self) -> Vec<usize> {
        self.rec.seq.audible().into_iter().map(|a| a.0).collect()
    }

    /// A timeline menu command, as the right-click menu sends it (tests).
    pub fn timeline_cmd(&mut self, c: crate::timelineui::Cmd) {
        self.menu_cmd(c);
    }

    /// A track header's mute / solo / gain, as the header sends it (tests).
    pub fn track_header(&mut self, t: usize, op: crate::timelineui::TrackOp) {
        self.track_op(t, op);
    }

    /// Select the cut at `at` on `track` as a roll edit point (tests).
    pub fn select_edit_at(&mut self, track: usize, at: i64) {
        self.rec.sel = crate::timelineui::Selection::Edit { track, at, side: model::Side::Roll };
    }

    /// A dissolve of `len` frames at the selected cut (tests).
    pub fn add_dissolve_len(&mut self, len: i64) {
        self.add_dissolve(Some(len));
    }

    pub fn pane_shown(&self, p: crate::panes::Pane) -> bool {
        self.panes.shown(p)
    }

    pub fn bin_width(&self) -> f32 {
        self.pane_rect(crate::panes::Pane::Bin).map_or(0.0, |r| r.width())
    }

    pub fn set_snap(&mut self, on: bool) {
        self.rec.snap = on;
    }

    pub fn seek_record(&mut self, f: u64) {
        self.rec.transport.stop();
        self.rec.transport.seek(f);
    }

    pub fn timeline_fpp(&self) -> f64 {
        self.rec.view.fpp()
    }

    /// Show a message in the record info strip.
    pub fn notice(&mut self, msg: String) {
        self.say(msg);
    }

    pub fn targets_audio(&self) -> bool {
        self.rec.targets.audio.is_some()
    }

    /// (bin, path) for every bin item, in order.
    pub fn bin_groups(&self) -> Vec<(String, PathBuf)> {
        self.bin.items.iter().map(|i| (i.group.clone(), i.path.clone())).collect()
    }

    /// Put a clip in the bin (tests).
    pub fn bin_add(&mut self, p: &Path) {
        self.bin.add(p, String::new());
    }

    /// The bin's tile view on / off (tests).
    pub fn bin_set_tiles(&mut self, on: bool) {
        self.bin.tiles = on;
        self.panes.bin_tiles = on;
        self.bin.prefs_in = true;
    }

    /// The bin's tiles on screen last frame (tests).
    pub fn bin_tile_rects(&self) -> Vec<(PathBuf, egui::Rect)> {
        self.bin.tile_rects.clone()
    }

    /// The tile scrubber: (frames, exact frame f held, frames held) (tests).
    pub fn bin_scrub(&self, path: &std::path::Path, f: u64) -> (Option<u64>, bool, usize) {
        match &self.bin.scrub {
            Some(s) => (s.info(path).map(|i| i.frames), s.has(path, f), s.held(path)),
            None => (None, false, 0),
        }
    }

    /// Hover-scrub latencies and decode times, ms (tests).
    pub fn bin_scrub_ms(&self) -> (Vec<f32>, Vec<f32>) {
        self.bin.scrub.as_ref().map_or_else(Default::default, |s| (s.latency.iter().copied().collect(), s.decode_ms.iter().copied().collect()))
    }

    pub fn bin_chosen(&self) -> usize {
        self.bin.chosen.len()
    }

    pub fn bin_scanning(&self) -> bool {
        self.bin.scanning()
    }

    pub fn bin_poll(&mut self) {
        self.bin.poll_scan();
    }

    pub fn sequence_has_audio(&self) -> bool {
        model::AUDIO_TRACKS.iter().any(|&t| !self.rec.seq.tracks[t].clips.is_empty())
    }

    pub fn sequence_fps(&self) -> f64 {
        self.rec.seq.fps.as_f64()
    }

    // --- editing -------------------------------------------------------------

    /// Insert (`insert`) or overwrite the source marks into the record
    /// marks on the targeted tracks. With all FOUR marks set the source
    /// in..out is fitted to the record in..out (its speed changes; the HUD
    /// says so); otherwise the three-point rules decide.
    pub(crate) fn edit(&mut self, insert: bool) {
        if self.source.is_none() {
            self.error = Some("load a clip into the source viewer first".into());
            return;
        }
        let media = self.media_of_source();
        let Some(c) = &self.source else { return };
        let r = &mut self.rec;
        let fps = Fps::new(c.info.fps_num, c.info.fps_den);
        // Conforming: a source frame is a record frame, so the marks
        // resolve as if the two rates were the same.
        let conform = r.seq.conform && !fps.same(r.seq.fps);
        let Some(res) = model::resolve(
            c.marks,
            c.transport.frame() as i64,
            c.info.frames as i64,
            fps,
            r.marks,
            r.transport.frame() as i64,
            if conform { fps } else { r.seq.fps },
        ) else {
            self.error = Some("nothing to edit: the source in is at or past its end".into());
            return;
        };
        let mut src = model::Source { conform, ..model::Source::new(media, res.src_in, res.len) };
        let mut fitted = None;
        let mut res = res;
        if let (Some(si), Some(so), Some(ri), Some(ro)) = (c.marks.inp, c.marks.out, r.marks.inp, r.marks.out) {
            let (sl, rl) = (so - si + 1, ro - ri + 1);
            if let Some(f) = r.seq.fit_source(media, si, sl, rl) {
                src = f;
                res = model::Resolved { src_in: si, rec_in: ri, len: rl };
                if (f.speed - 1.0).abs() > 1e-9 {
                    fitted = Some((sl, rl, f.speed));
                }
            }
        }
        if insert {
            r.seq.insert(src, res.rec_in, r.targets);
        } else {
            r.seq.overwrite(src, res.rec_in, r.targets);
        }
        if !self.api_quiet {
            r.history.touched();
            r.history.commit(&r.seq);
        }
        // As every NLE does: the record playhead lands after the edit and
        // the record marks are used up; the source marks stay.
        r.marks.clear();
        r.fit_transport();
        r.transport.seek((res.rec_in + res.len) as u64);
        self.error = None;
        if let Some((sl, rl, speed)) = fitted {
            self.say(format!("fit to fill: {sl} source frames over {rl} record frames, speed {speed:.3}x"));
        }
    }

    /// M: append. The source in..out (the whole clip with no marks) goes
    /// on the end of the sequence, after its last clip, on the targeted
    /// tracks with its linked sound: an overwrite whose record in is the
    /// end. One undo step; the playhead lands at the new end.
    pub(crate) fn append(&mut self) {
        let end = self.rec.seq.duration().max(0);
        let saved = self.rec.marks;
        self.rec.marks = Marks { inp: Some(end), out: None };
        self.edit(false);
        if self.error.is_some() {
            self.rec.marks = saved;
        } else {
            self.say("appended at the end (M)");
        }
    }

    pub(crate) fn undo_public(&mut self) {
        self.undo(false);
    }

    pub(crate) fn undo_public_dir(&mut self, redo: bool) {
        self.undo(redo);
    }

    fn undo(&mut self, redo: bool) {
        let r = &mut self.rec;
        let changed = if redo { r.history.redo(&mut r.seq) } else { r.history.undo(&mut r.seq) };
        if changed {
            r.fit_transport();
            r.transport.seek(r.transport.frame());
        }
    }

    // --- keys ------------------------------------------------------------------

    fn keys(&mut self, ui: &mut egui::Ui) {
        // Esc closes the text window first, whether its text has focus or not.
        if self.textv.open.is_some() && take(ui, Modifiers::NONE, Key::Escape, false) {
            self.textv.open = None;
            ui.ctx().memory_mut(|m| m.stop_text_input());
            self.changed(false);
            return;
        }
        // Over the clip browser window its own keys rule (type-ahead,
        // arrows, Enter, Space, Esc).
        #[cfg(feature = "clips")]
        if self.over_clips(ui) {
            self.hold = None;
            return;
        }
        if ui.ctx().text_edit_focused() {
            // Letters typed into a field are never holds, even once the
            // field lets go and they are still down.
            self.hold = None;
            self.hold_chord = true;
            return;
        }
        let none = Modifiers::NONE;
        let cmd = Modifiers::COMMAND;
        let shift = Modifiers::SHIFT;
        let alt = Modifiers::ALT;
        // EXPERIMENT (fold): cmd shift W (handled in `ui`) makes the selected
        // clip a wob and dives into wobble, and comes back out; cmd alt W
        // gives the clip its footage back.
        #[cfg(feature = "fold")]
        {
            if take(ui, cmd | alt, Key::W, false) {
                self.fold_clear();
            }
            // Enter on a wob clip dives into it.
            if self.fold_selected_is_wob() && take(ui, none, Key::Enter, false) {
                self.fold_make();
            }
        }
        // Panes: Q alone / back, W hide / show, shift+W all back, H key hints.
        // (Q is the dock's: its solo key.)
        let ptr = ui.input(|i| i.pointer.hover_pos());
        if take(ui, none, Key::Backtick, false) {
            self.toggle_mode();
        }
        if self.mode == Mode::Edit && take(ui, none, Key::W, false) {
            if let Some((p, _)) = ptr.and_then(|q| self.panes.at(q)) {
                self.set_pane_open(p, false);
            }
        }
        if take(ui, shift, Key::W, false) {
            self.show_all_panes();
        }
        if take(ui, shift, Key::H, false) {
            let on = !self.panes.readouts;
            self.panes.set_readouts(on);
            self.say(if on { "viewer readouts on (shift H)" } else { "viewer readouts off (shift H)" });
        }
        // R over a viewer frames it; shift+R matches the other viewer to it
        // (same size, same zoom and pan). R over the timeline fits it.
        let over = ui.input(|i| i.pointer.hover_pos()).and_then(|q| self.panes.at(q)).filter(|(_, strip)| !strip).map(|(p, _)| p);
        let viewer = match over {
            Some(crate::panes::Pane::Source) => Some(0),
            Some(crate::panes::Pane::Record) => Some(1),
            _ => None,
        };
        if over == Some(crate::panes::Pane::Timeline) && take(ui, none, Key::R, false) {
            self.rec.view.fit();
            self.say("timeline fitted (R)");
        }
        if let Some(i) = viewer.filter(|_| self.mode == Mode::Edit) {
            if take(ui, none, Key::R, false) {
                self.views[i].fit();
            }
            if take(ui, shift, Key::R, false) {
                self.views[1 - i] = self.views[i];
                self.match_viewers();
                self.say("both viewers: same size and view");
            }
        }
        if take(ui, none, Key::H, false) {
            let on = !self.panes.hints;
            self.panes.set_hints(on);
        }
        // An open popup (the timeline's right-click menu, a menubar menu)
        // takes Esc itself: never steal it.
        let popup = egui::Popup::is_any_open(ui.ctx());
        if !popup && self.panes.solo.is_some() && self.drag.is_none() && take(ui, none, Key::Escape, false) {
            let h = self.panes.host;
            if let Some(d) = &mut self.docks {
                d.unsolo(h);
            }
            self.panes.solo = None;
        }
        if take(ui, cmd, Key::I, false) {
            self.import();
        }
        if take(ui, cmd, Key::O, false) {
            self.open_project_dialog();
        }
        if take(ui, cmd, Key::S, false) {
            self.save(false);
        }
        if take(ui, cmd, Key::M, false) {
            self.ask_render();
        }
        if take(ui, cmd | shift, Key::R, false) {
            let ctx = ui.ctx().clone();
            self.toggle_recording(&ctx);
        }
        if take(ui, cmd | shift, Key::S, false) {
            self.save(true);
        }
        if take(ui, cmd, Key::Q, false) && self.request_quit() {
            self.quit = true;
        }
        if take(ui, cmd, Key::T, false) {
            self.opt.ask();
        }
        if take(ui, cmd, Key::Z, false) {
            self.undo(false);
        }
        if take(ui, cmd | shift, Key::Z, false) {
            self.undo(true);
        }
        for (k, f) in [(Key::Num1, Focus::Bin), (Key::Num2, Focus::Source), (Key::Num3, Focus::Record)] {
            if take(ui, shift, k, false) {
                self.focus = f;
                if f != Focus::Bin {
                    self.monitor = f;
                }
            }
        }
        if self.mode == Mode::Selects {
            self.selects_keys(ui);
            self.mark_keys(ui);
            self.transport_keys(ui);
            return;
        }
        if take(ui, none, Key::Comma, false) {
            self.edit(true);
        }
        if take(ui, none, Key::Period, false) {
            self.edit(false);
        }
        if take(ui, none, Key::M, false) {
            // With the bin focused: its chosen clips and subclips, in order.
            if self.focus == Focus::Bin {
                self.append_chosen();
            } else {
                self.append();
            }
        }
        for (k, t) in [
            (Key::Num1, model::V1),
            (Key::Num2, model::V2),
            (Key::Num3, model::V3),
            (Key::Num4, model::A1),
            (Key::Num5, model::A2),
            (Key::Num6, model::A3),
            (Key::Num7, model::M),
        ] {
            if take(ui, none, k, false) {
                self.rec.targets.toggle(t);
            }
        }
        if take(ui, none, Key::Equals, true) || take(ui, shift, Key::Equals, true) || take(ui, shift, Key::Plus, true) {
            self.rec.view.zoom(1.25);
        }
        if take(ui, none, Key::Minus, true) {
            self.rec.view.zoom(0.8);
        }
        if take(ui, none, Key::Backslash, false) {
            self.rec.view.fit();
        }

        // Step 3: clips, edit points, trims, slip, speed, dissolves.
        if self.focus != Focus::Bin {
            // Hold keys (quasimodes, `mouse::HoldOp`): while A S D Z C (or
            // shift A S D) is held, a press anywhere on a clip drags that
            // op. A held key that dragged fires nothing on release; D
            // tapped alone still selects the clip under the playhead.
            const HOLD_KEYS: [Key; 5] = [Key::A, Key::S, Key::D, Key::Z, Key::C];
            let (held_key, pressed_d, released_d, pressed_shift, live_shift) = ui.input(|i| {
                let plain = |m: &Modifiers| !m.command && !m.alt && !m.ctrl;
                let held = HOLD_KEYS.into_iter().find(|k| i.key_down(*k));
                let pd = i.events.iter().find_map(|e| match e {
                    egui::Event::Key { key: Key::D, pressed: true, repeat: false, modifiers, .. } if plain(modifiers) => Some(modifiers.shift),
                    _ => None,
                });
                let rel = i.events.iter().any(|e| matches!(e, egui::Event::Key { key: Key::D, pressed: false, .. }));
                // Shift as the hold key went down counts, with the live state.
                // (shift, and whether cmd / alt / ctrl came with it: then it
                // is a chord like cmd+A, not a hold).
                let ps = i.events.iter().find_map(|e| match e {
                    egui::Event::Key { key, pressed: true, modifiers, .. } if HOLD_KEYS.contains(key) => Some((modifiers.shift, !plain(modifiers))),
                    _ => None,
                });
                (held, pd, rel, ps, i.modifiers.shift)
            });
            if let Some((sh, chord)) = pressed_shift {
                self.hold_shift = sh;
                self.hold_chord = chord;
            }
            let any_down = held_key.is_some();
            let hold = held_key.filter(|_| !self.hold_chord).and_then(|k| crate::mouse::HoldOp::of(k, self.hold_shift || live_shift));
            if let Some(shift_d) = pressed_d {
                self.d_tap = Some(shift_d);
            }
            if !(ui.input(|i| i.modifiers.command || i.modifiers.alt)) {
                self.hold = hold;
            } else {
                self.hold = None;
            }
            if released_d {
                if let (Some(shift_d), false) = (self.d_tap.take(), self.hold_used) {
                    self.select_clip(shift_d);
                }
            }
            if !any_down && self.drag.is_none() {
                self.hold_used = false;
                self.hold_chord = false;
            }
            for k in HOLD_KEYS {
                take(ui, none, k, false);
                take(ui, shift, k, false);
            }
            if take(ui, none, Key::G, false) {
                self.select_edit();
            }
            if !popup && take(ui, none, Key::Escape, false) && !self.drag_cancel() {
                self.rec.sel = crate::timelineui::Selection::None;
            }
            if take(ui, none, Key::T, false) {
                self.rec.snap = !self.rec.snap;
                let m = if self.rec.snap { "snapping on (T)" } else { "snapping off (T)" };
                self.say(m);
            }
            // Tools (the razor is X: R fits the viewer or timeline under the
            // pointer, above). The rate
            // stretch tool has no key now (shift A / shift D hold-drags do
            // it); it stays in sequence > tools.
            for (k, t) in [
                (Key::V, crate::timelineui::Tool::Select),
                (Key::X, crate::timelineui::Tool::Razor),
                (Key::Y, crate::timelineui::Tool::Slip),
                (Key::U, crate::timelineui::Tool::Slide),
                (Key::N, crate::timelineui::Tool::Roll),
            ] {
                if take(ui, none, k, false) {
                    self.set_tool(t);
                }
            }
            if take(ui, cmd, Key::K, false) {
                self.razor_playhead(false);
            }
            if take(ui, cmd | shift, Key::K, false) {
                self.razor_playhead(true);
            }
            if take(ui, none, Key::B, false) {
                self.rec.ripple = !self.rec.ripple;
                let m = if self.rec.ripple { "ripple trims (yellow)" } else { "plain trims (red)" };
                self.say(m);
            }
            if take(ui, none, Key::E, false) {
                self.extend_edit();
            }
            for (k, d) in [(Key::ArrowLeft, -1), (Key::ArrowRight, 1)] {
                if take(ui, alt, k, true) {
                    self.trim_by(d);
                }
                if take(ui, alt | shift, k, true) {
                    self.trim_by(d * 10);
                }
            }
            for (k, d) in [(Key::Comma, -1), (Key::Period, 1)] {
                if take(ui, alt, k, true) {
                    self.slip_by(d);
                }
                if take(ui, alt | shift, k, true) {
                    self.slip_by(d * 10);
                }
            }
            for k in [Key::Delete, Key::Backspace] {
                if take(ui, none, k, false) {
                    self.delete_clip(false);
                }
                if take(ui, shift, k, false) {
                    self.delete_clip(true);
                }
            }
            if take(ui, cmd, Key::R, false) {
                self.ask_speed();
            }
            if take(ui, alt, Key::R, false) {
                self.toggle_conform();
            }
            if take(ui, shift, Key::Period, false) {
                self.fit_to_fill();
            }
            if take(ui, cmd, Key::D, false) {
                self.add_dissolve(None);
            }
            if take(ui, cmd | shift, Key::D, false) {
                self.remove_dissolve();
            }
            if take(ui, alt, Key::Equals, true) {
                self.dissolve_by(2);
            }
            // Reframe the selected clip in the raster.
            if take(ui, none, Key::F, false) {
                self.reframe_with(|r| r.fill = !r.fill);
            }
            if take(ui, alt, Key::F, false) {
                self.reframe_with(|r| *r = model::Reframe::default());
            }
            for (k, z) in [(Key::OpenBracket, 1.0 / 1.05), (Key::CloseBracket, 1.05)] {
                if take(ui, none, k, true) {
                    self.reframe_with(|r| r.zoom *= z);
                }
            }
            for (k, z) in [(Key::OpenBracket, 1.0 / 1.01), (Key::CloseBracket, 1.01)] {
                if take(ui, shift, k, true) {
                    self.reframe_with(|r| r.zoom *= z);
                }
            }
            // cmd+arrows: a text clip's style moves; any other clip reframes.
            for (k, dx, dy) in [(Key::ArrowLeft, -1.0, 0.0), (Key::ArrowRight, 1.0, 0.0), (Key::ArrowUp, 0.0, -1.0), (Key::ArrowDown, 0.0, 1.0)] {
                if take(ui, cmd, k, true) && !self.nudge_text(dx as f32 * 0.01, dy as f32 * 0.01) {
                    self.reframe_with(|r| (r.x, r.y) = (r.x + dx * 0.01, r.y + dy * 0.01));
                }
                if take(ui, cmd | shift, k, true) && !self.nudge_text(dx as f32 * 0.1, dy as f32 * 0.1) {
                    self.reframe_with(|r| (r.x, r.y) = (r.x + dx * 0.1, r.y + dy * 0.1));
                }
            }
            #[cfg(feature = "wobble")]
            if take(ui, cmd | shift, Key::W, false) {
                self.wobble_open();
            }
            // Titles: cmd+shift+T makes one; Enter edits the selected one.
            if take(ui, cmd | shift, Key::T, false) {
                self.new_text(None, None, "Title");
            }
            if let Some((t, i)) = self.selected_text() {
                if take(ui, none, Key::Enter, false) {
                    let st = self.rec.seq.tracks[t].clips[i].start;
                    self.open_text(t, st);
                }
            }
            if take(ui, alt, Key::Minus, true) {
                self.dissolve_by(-2);
            }
        }

        if self.focus == Focus::Bin {
            if let bin::Action::Load(p, r) = self.bin.keys(ui) {
                self.load_item(&p, r);
                self.focus = Focus::Bin;
            }
        } else {
            // Edit points walk the record playhead from anywhere but the bin.
            let t = self.rec.transport.frame() as i64;
            let up = take(ui, none, Key::ArrowUp, true).then(|| self.rec.seq.prev_edit(t)).flatten();
            let down = take(ui, none, Key::ArrowDown, true).then(|| self.rec.seq.next_edit(t)).flatten();
            if let Some(p) = up.or(down) {
                self.rec.transport.stop();
                self.rec.fit_transport();
                self.rec.transport.seek(p as u64);
                self.focus = Focus::Record;
                self.monitor = Focus::Record;
            }
        }

        self.mark_keys(ui);
        self.transport_keys(ui);
    }

    /// I O and friends: the side under the pointer (the source viewer, or
    /// the record viewer / timeline), else the focused one.
    fn mark_keys(&mut self, ui: &mut egui::Ui) {
        let (none, shift, alt) = (Modifiers::NONE, Modifiers::SHIFT, Modifiers::ALT);
        // Marks: the side under the pointer (the source viewer, or the
        // record viewer / timeline), else the focused one.
        let mark_side = match self.key_target(ui) {
            Some(crate::panes::Pane::Source) if self.source.is_some() => Focus::Source,
            Some(crate::panes::Pane::Record | crate::panes::Pane::Timeline) => Focus::Record,
            _ => self.focus,
        };
        let side = match mark_side {
            Focus::Record => Some((&mut self.rec.marks, &mut self.rec.transport)),
            Focus::Source => self.source.as_mut().map(|c| (&mut c.marks, &mut c.transport)),
            Focus::Bin => None,
        };
        if let Some((marks, t)) = side {
            let f = t.frame() as i64;
            if take(ui, none, Key::I, false) {
                marks.set_in(f);
            }
            if take(ui, none, Key::O, false) {
                marks.set_out(f);
            }
            if take(ui, shift, Key::I, false) {
                if let Some(i) = marks.inp {
                    t.stop();
                    t.seek(i as u64);
                }
            }
            if take(ui, shift, Key::O, false) {
                if let Some(o) = marks.out {
                    t.stop();
                    t.seek(o as u64);
                }
            }
            if take(ui, alt, Key::I, false) {
                marks.inp = None;
            }
            if take(ui, alt, Key::O, false) {
                marks.out = None;
            }
            if take(ui, alt, Key::X, false) {
                marks.clear();
            }
        }
    }

    /// J K L, space, steps, home / end: the active viewer.
    fn transport_keys(&mut self, ui: &mut egui::Ui) {
        let bin_focus = self.focus == Focus::Bin;
        let none = Modifiers::NONE;
        let Some(t) = self.active_transport() else { return };
        if take(ui, none, Key::L, false) {
            t.shuttle_forward();
        }
        if take(ui, none, Key::J, false) {
            t.shuttle_back();
        }
        if take(ui, none, Key::K, false) {
            t.stop();
        }
        if take(ui, none, Key::Space, false) {
            t.toggle();
        }
        if !bin_focus {
            if take(ui, none, Key::ArrowLeft, true) {
                t.step_by(-1);
            }
            if take(ui, none, Key::ArrowRight, true) {
                t.step_by(1);
            }
        }
        if take(ui, none, Key::Home, false) {
            t.stop();
            t.seek(0);
        }
        if take(ui, none, Key::End, false) {
            t.stop();
            t.seek(t.last());
        }
    }

    // --- sound --------------------------------------------------------------------

    /// What should be sounding now: the playing viewer's audio at 1x. Starts,
    /// restarts (an edit, a loop) and stops the output to match, and while
    /// it plays puts that viewer's playhead on the audio clock. Returns the
    /// viewer the clock is driving.
    fn sync_audio(&mut self) -> Option<Focus> {
        use crate::audio::{Plan, PlanClip};
        let want: Option<(Plan, f64, Fps, Focus)> = match self.monitor {
            Focus::Record => {
                let t = &self.rec.transport;
                (t.speed == 1.0 && !t.prerolling()).then(|| {
                    let seq = &self.rec.seq;
                    let prog = &self.rec.program;
                    let plan = crate::audio::plan_of(seq, &|m| prog.read_path(&seq.media[m].path).map(|p| p.to_path_buf()));
                    (plan, t.pos, seq.fps, Focus::Record)
                })
            }
            _ => self.source.as_ref().and_then(|c| {
                let t = &c.transport;
                (t.speed == 1.0 && !t.prerolling() && c.info.has_audio).then(|| {
                    let fps = Fps::new(c.info.fps_num, c.info.fps_den);
                    let clip = PlanClip {
                        path: c.server.source().info().path.clone(),
                        start: 0,
                        len: c.info.frames as i64,
                        src_in: 0,
                        src_fps: fps,
                        speed: 1.0,
                        gain: 1.0,
                        lane: 0,
                    };
                    (Plan { clips: vec![clip], fps }, t.pos, fps, Focus::Source)
                })
            }),
        };
        let want = want.filter(|w| w.0.clips.iter().any(|c| c.speed == 1.0));
        let Some((plan, from, fps, who)) = want else {
            if let Some(Ok(a)) = &mut self.audio {
                if a.is_playing() {
                    a.stop();
                }
            }
            self.audio_set = None;
            return None;
        };
        if self.audio.is_none() {
            self.audio = Some(crate::audio::Audio::open());
        }
        let audio = match &mut self.audio {
            Some(Ok(a)) => a,
            _ => return None,
        };
        if self.audio_off_until.is_some_and(|t| Instant::now() < t) {
            return None;
        }
        // A new plan, or the playhead moved from where the sound put it (a
        // seek while playing): the sound starts again from the playhead.
        let seeked = self.audio_set.is_some_and(|p| (p - from).abs() > 1e-6);
        if audio.plan() != Some(&plan) || seeked {
            audio.start(plan, from);
            self.audio_set = Some(from);
            return Some(who);
        }
        if audio.stalled() {
            // The device is not taking samples: never let the picture wait
            // on it. Play on the wall clock, silent, and try again later.
            audio.stop();
            self.audio_off_until = Some(Instant::now() + std::time::Duration::from_secs(3));
            self.say("the audio device stopped responding: playing without sound");
            return None;
        }
        let clock = audio.clock(fps)?;
        let t = match who {
            Focus::Record => &mut self.rec.transport,
            _ => &mut self.source.as_mut()?.transport,
        };
        let stop_at = match t.end {
            Some(e) if !t.looping && t.pos < e as f64 => e as f64,
            _ => t.frames as f64,
        };
        if clock >= stop_at {
            // Off the end: the transport loops or stops, the sound follows.
            t.tick(((clock - t.pos) / (t.fps * t.speed)).max(0.0));
            audio.stop();
            return Some(who);
        }
        t.pos = clock;
        self.audio_set = Some(clock);
        Some(who)
    }

    // --- feeding the decoders ----------------------------------------------------

    /// A transport that just started holds until its first frames are in.
    fn preroll(&mut self) {
        if let Some(c) = &mut self.source {
            if let Some(fs) = c.transport.preroll_frames(DISPLAY_HZ) {
                let n = c.info.frames as i64;
                let ready = fs.iter().filter(|&&f| (0..n).contains(&f)).all(|&f| c.server.is_cached(f as u64));
                c.transport.primed(ready);
            }
        }
        let r = &mut self.rec;
        if let Some(fs) = r.transport.preroll_frames(DISPLAY_HZ) {
            let ready = fs.iter().all(|&t| r.program.is_cached(&r.seq, t));
            r.transport.primed(ready);
        }
    }

    /// Tell the servers what to have ready, after this redraw's keys and scrubs.
    fn feed(&mut self) {
        if let Some(c) = &self.source {
            c.server.set(play_state(&c.transport));
        }
        let extra: Vec<model::Layer> = self.trim_frames().map(|(a, b)| a.into_iter().chain(b).collect()).unwrap_or_default();
        let r = &mut self.rec;
        r.program.sync(&r.seq, &self.opt.settings);
        let st = match r.peek {
            Some(p) => PlayState::paused(p.max(0) as u64),
            None => play_state(&r.transport),
        };
        r.program.feed(&r.seq, st, &extra);
    }

    // --- drawing ---------------------------------------------------------------------

    /// A viewer's title bar; on the right a "clear in/out" button (enabled
    /// while that side has a mark), the key hints' on / off button and the
    /// zoom when not fitted. (clear clicked, keys clicked).
    fn title_bar(&self, ui: &mut egui::Ui, rect: Rect, text: &str, focused: bool, marks: bool, zoom: f32) -> (bool, bool) {
        let v = ui.visuals().clone();
        let p = ui.painter_at(rect);
        let fill = if focused { v.selection.bg_fill } else { v.widgets.noninteractive.bg_fill };
        p.rect_filled(rect, 0.0, fill);
        // The dock's grip leads the bar: drag the viewer by it, right-click
        // for its menu.
        let gw = ui
            .scope_builder(egui::UiBuilder::new().max_rect(rect.shrink2(vec2(2.0, 0.0))), |ui| ui.horizontal_centered(|ui| imfram::dock::grip(ui)).response.rect.width())
            .inner;
        let col = if focused { v.strong_text_color() } else { v.text_color() };
        p.text(rect.left_center() + vec2(8.0 + gw, 0.0), Align2::LEFT_CENTER, text, FontId::proportional(12.0), col);
        // Clear in/out: plain to see whenever this side has a mark.
        let b = if marks { Rect::from_min_max(pos2(rect.max.x - 124.0, rect.min.y + 1.0), pos2(rect.max.x - 3.0, rect.max.y - 1.0)) } else { Rect::from_min_max(rect.right_top(), rect.right_bottom()) };
        let clicked = marks
            && ui
                .put(b, egui::Button::new(egui::RichText::new("× clear in/out").size(12.0).strong().color(Color32::WHITE)).fill(CLEAR_MARKS))
                .on_hover_text("clear the in and out marks (alt X; alt I / alt O: one of them)")
                .clicked();
        // The key hints over the viewers, on / off (H).
        let k = Rect::from_min_max(pos2(b.min.x - 4.0 - 40.0, rect.min.y + 2.0), pos2(b.min.x - 4.0, rect.max.y - 2.0));
        let keys = ui
            .put(k, egui::Button::new(egui::RichText::new("keys").size(11.0)).selected(self.panes.hints))
            .on_hover_text(if self.panes.hints { "hide the key hints  (H)" } else { "show the key hints  (H)" })
            .clicked();
        if (zoom - 1.0).abs() > 1e-3 {
            p.text(pos2(k.min.x - 8.0, rect.center().y), Align2::RIGHT_CENTER, format!("{:.0}%  (R: fit)", zoom * 100.0), FontId::proportional(11.0), col);
        }
        (clicked, keys)
    }

    /// RV-style scrub by dragging viewer `slot`'s picture: from the frame
    /// on screen at the press, one frame per `SCRUB_PX` points of
    /// horizontal travel (either way). `start_ok`: a drag may begin here.
    fn picture_scrub(&mut self, ui: &egui::Ui, slot: usize, hand: &egui::Response, navigated: bool, start_ok: bool, now: i64) -> Option<i64> {
        let m = ui.input(|i| i.modifiers) | self.press_mods;
        let plain = !m.shift && !m.alt && !m.command;
        if start_ok && plain && !navigated && hand.drag_started_by(egui::PointerButton::Primary) {
            let x0 = ui.input(|i| i.pointer.press_origin()).map_or(0.0, |p| p.x);
            self.view_scrub[slot] = Some((now, x0));
        }
        let (f0, x0) = self.view_scrub[slot]?;
        if !ui.input(|i| i.pointer.primary_down()) {
            self.view_scrub[slot] = None;
            return None;
        }
        let x = ui.input(|i| i.pointer.latest_pos()).map_or(x0, |p| p.x);
        Some(f0 + ((x - x0) / SCRUB_PX).round() as i64)
    }

    /// A file read as `space` (`None`: by its format), undoable.
    pub(crate) fn set_input_space(&mut self, path: &Path, space: Option<colour::color::InputSpace>) {
        let c = &mut self.rec.seq.colour;
        match space {
            Some(s) => {
                c.inputs.insert(path.to_path_buf(), s);
            }
            None => {
                c.inputs.remove(path);
            }
        }
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let l = self.rec.seq.colour.input_of(path).label();
        self.changed(false);
        self.say(format!("{name} read as {l}"));
    }

    /// The colour menu: the project's working space, the view (transform,
    /// display, curve, look, exposure, gamma) and the source clip's input.
    fn colour_menu(&mut self, ui: &mut egui::Ui) {
        use colour::color::Colorspace;
        use colour::viewtf::{Eotf, Gamut, ViewPreset, ViewTransform};
        let mut c = self.rec.seq.colour.clone();
        ui.menu_button(format!("working space: {}", c.working.label()), |ui| {
            // wobble's list: every scene-linear preset.
            for (name, cs) in Colorspace::PRESETS.into_iter().filter(|(_, cs)| cs.is_scene_linear()) {
                if ui.radio(c.working == cs, name).clicked() {
                    c.working = cs;
                }
            }
        });
        ui.separator();
        // wobble's viewer menu: a transform on a display, by name.
        ui.menu_button(format!("view preset: {}", c.view.preset().label()), |ui| {
            let mut aces12 = false;
            for (name, p) in ViewPreset::CATALOGUE {
                let is_12 = p.transform == ViewTransform::Aces12;
                if is_12 && !aces12 {
                    ui.weak("ACES 1.2");
                }
                aces12 |= is_12;
                let r = ui.add_enabled(p.transform.available(), egui::RadioButton::new(c.view.preset() == p, name));
                if r.clicked() {
                    c.view = c.view.with_preset(p);
                }
            }
        });
        ui.menu_button(format!("view: {}", c.view.transform.label()), |ui| {
            for t in ViewTransform::ALL {
                let r = ui.add_enabled(t.available(), egui::RadioButton::new(c.view.transform == t, t.label()));
                if r.clicked() {
                    c.view.transform = t;
                }
            }
        });
        ui.menu_button(format!("display: {}", c.view.display.label()), |ui| {
            for g in Gamut::DISPLAYS {
                if ui.radio(c.view.display == g, g.label()).clicked() {
                    c.view.display = g;
                }
            }
        });
        ui.menu_button(format!("curve: {}", c.view.eotf.label()), |ui| {
            for e in Eotf::ALL {
                if ui.radio(c.view.eotf == e, e.label()).clicked() {
                    c.view.eotf = e;
                }
            }
        });
        let looks = c.view.effective().looks();
        if looks.len() > 1 {
            ui.menu_button(format!("look: {}", c.view.look.label()), |ui| {
                for &l in looks {
                    if ui.radio(c.view.look == l, l.label()).clicked() {
                        c.view.look = l;
                    }
                }
            });
        }
        ui.add(egui::Slider::new(&mut c.view.exposure, -6.0..=6.0).step_by(0.1).text("exposure (stops)"));
        ui.add(egui::Slider::new(&mut c.view.gamma, 0.2..=4.0).step_by(0.05).text("gamma"));
        if ui.button("exposure 0, gamma 1").clicked() {
            c.view.exposure = 0.0;
            c.view.gamma = 1.0;
        }
        ui.separator();
        if let Some(p) = self.source.as_ref().map(|s| s.original.clone()) {
            let mut pick = None;
            ui.menu_button(format!("source clip read as: {}", c.input_of(&p).label()), |ui| {
                pick = input_space_menu(ui, &c, &p);
            });
            if let Some(sp) = pick {
                self.rec.seq.colour = c;
                self.set_input_space(&p, sp);
                return;
            }
        }
        ui.weak("a timeline clip: right-click it > input colour");
        if c != self.rec.seq.colour {
            self.rec.seq.colour = c;
            self.changed(false);
        }
    }

    /// Once a frame: the recorder's readbacks collected, its review window
    /// drawn, its options kept; the screenshot's file written.
    fn recorder_tick(&mut self, ctx: &egui::Context) {
        let Some(device) = self.device.clone() else { return };
        if let Ok(mut r) = self.recorder.lock() {
            if r.poll(&device) {
                r.stop(&device);
            }
            if r.recording() {
                ctx.request_repaint();
            }
            r.ui(ctx, None);
            if std::mem::take(&mut r.opts_dirty) {
                self.panes.rec_opts = Some(r.opts.clone());
                self.panes.changed();
            }
        }
        if let Ok(mut s) = self.shot.lock() {
            if s.poll(&device) || s.busy() {
                ctx.request_repaint();
            }
        }
    }

    /// The panel that wore the key-target edge last (tests).
    pub fn key_panel(&self) -> Option<crate::panes::Pane> {
        self.key_panel
    }

    pub fn set_source_marks(&mut self, m: Marks) {
        if let Some(c) = &mut self.source {
            c.marks = m;
        }
    }

    pub fn focus_bin(&mut self) {
        self.focus = Focus::Bin;
    }

    pub fn focus_source(&mut self) {
        self.focus = Focus::Source;
    }

    /// The torn-off windows' panel rects (tests).
    pub fn pane_rect_in(&self, host: usize, p: crate::panes::Pane) -> Option<Rect> {
        self.panes.in_host(host).into_iter().find(|(q, _)| *q == p).map(|(_, r)| r)
    }

    /// The timeline shows the whole sequence (tests).
    pub fn timeline_fitted(&self) -> bool {
        self.rec.view.fitted()
    }

    /// Zoom the timeline (tests).
    pub fn timeline_zoom(&mut self, f: f32) {
        self.rec.view.zoom(f);
    }

    /// The source playhead (tests).
    pub fn source_frame(&self) -> Option<i64> {
        self.source.as_ref().map(|c| c.transport.frame() as i64)
    }

    /// The source marks (tests).
    pub fn source_marks(&self) -> Option<Marks> {
        self.source.as_ref().map(|c| c.marks)
    }

    /// The record playhead (tests).
    pub fn record_frame(&self) -> i64 {
        self.rec.transport.frame() as i64
    }

    /// The active viewer's playhead, fractional (tests, `--play`).
    pub fn playhead_pos(&self) -> f64 {
        match self.monitor {
            Focus::Record => self.rec.transport.pos,
            _ => self.source.as_ref().map_or(0.0, |c| c.transport.pos),
        }
    }

    /// The active viewer is still pre-rolling (holding for its first frames).
    pub fn prerolling(&self) -> bool {
        match self.monitor {
            Focus::Record => self.rec.transport.prerolling(),
            _ => self.source.as_ref().is_some_and(|c| c.transport.prerolling()),
        }
    }

    /// The active viewer's frame rate.
    pub fn active_fps(&self) -> f64 {
        match self.monitor {
            Focus::Record => self.rec.seq.fps.as_f64(),
            _ => self.source.as_ref().map_or(24.0, |c| c.info.fps()),
        }
    }

    /// The record sequence (tests).
    pub fn sequence(&self) -> &model::Sequence {
        &self.rec.seq
    }

    /// The project's colour settings (tests, scripts).
    pub fn colour_mut(&mut self) -> &mut model::ColourSettings {
        &mut self.rec.seq.colour
    }

    /// Where viewer `slot`'s picture was drawn last frame (tests).
    pub fn picture_rect(&self, slot: usize) -> Option<Rect> {
        self.pic_rects.get()[slot.min(1)]
    }

    /// A file's input transform into the project's working space.
    pub(crate) fn idt_for(&self, path: &Path) -> Arc<crate::viewer::Idt> {
        let c = &self.rec.seq.colour;
        let (src, w) = (c.input_of(path), c.working);
        let k = format!("{src:?}>{w:?}");
        self.idts.borrow_mut().entry(k).or_insert_with(|| Arc::new(crate::viewer::Idt::new(src, w))).clone()
    }

    /// The view the viewers show through, its working primaries mirrored
    /// from the project's working space.
    pub(crate) fn view_settings(&self) -> colour::viewtf::ViewSettings {
        let mut v = self.rec.seq.colour.view;
        v.mirror_working(self.rec.seq.colour.working);
        v
    }

    #[allow(clippy::too_many_arguments)]
    fn picture(ui: &egui::Ui, rect: Rect, slot: usize, dims: (u32, u32), frame: Option<(u64, media::FrameRef)>, black: bool, view: crate::viewnav::View, idt: Arc<crate::viewer::Idt>, vs: colour::viewtf::ViewSettings) {
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::BLACK);
        let (w, h) = (dims.0.max(1) as f32, dims.1.max(1) as f32);
        let scale = (rect.width() / w).min(rect.height() / h) * view.zoom;
        let img = view.apply(rect, Rect::from_center_size(rect.center(), vec2(w, h) * (scale / view.zoom)));
        let nearest = scale * ui.ctx().pixels_per_point() > 1.0;
        painter.add(egui_wgpu::Callback::new_paint_callback(img, ViewerPaint { slot, frame, idt, view: vs, black, nearest, xf: [0.0, 0.0, 1.0, 1.0], mix: None, ext: None }));
    }

    /// The record picture: drawn over the whole raster rect, each picture
    /// placed by its clip's reframe (black outside, clipped to the raster).
    fn raster_picture(
        ui: &egui::Ui,
        raster_rect: Rect,
        raster: (u32, u32),
        a: Option<((u64, media::FrameRef), model::Reframe)>,
        dims: (u32, u32),
        reframe: model::Reframe,
        black: bool,
        mix: Option<(u64, media::FrameRef, f32, model::Reframe, Arc<crate::viewer::Idt>)>,
        idt: Arc<crate::viewer::Idt>,
        vs: colour::viewtf::ViewSettings,
        ext: Option<(wgpu::TextureView, (u32, u32), Arc<crate::viewer::Idt>)>,
    ) {
        let painter = ui.painter_at(raster_rect);
        painter.rect_filled(raster_rect, 0.0, Color32::BLACK);
        // A wob clip: the comp's own size places it, and it is never black.
        let (dims, black) = match &ext {
            Some((_, d, _)) => (*d, false),
            None => (dims, black),
        };
        let xf = |r: model::Reframe, d: (u32, u32)| {
            let (x, y, w, h) = r.place(d, raster);
            let (rw, rh) = (raster.0.max(1) as f64, raster.1.max(1) as f64);
            [(x / rw) as f32, (y / rh) as f32, (w / rw) as f32, (h / rh) as f32]
        };
        let (x, _, w, _) = reframe.place(dims, raster);
        let _ = x;
        let px_per_src = raster_rect.width() as f64 / raster.0.max(1) as f64 * w / dims.0.max(1) as f64;
        let nearest = px_per_src * ui.ctx().pixels_per_point() as f64 > 1.0;
        let xa = xf(reframe, dims);
        let frame = a.map(|(f, _)| f);
        let mix = mix.map(|(k, f, m, r, bi)| {
            let d = (f.width, f.height);
            (2usize, k, f, bi, m, xf(r, d))
        });
        painter.add(egui_wgpu::Callback::new_paint_callback(raster_rect, ViewerPaint { slot: 1, frame, idt, view: vs, black, nearest, xf: xa, mix, ext }));
    }

    /// Network storage that cannot keep up, said in words over the picture
    /// while playing (never slow silently): the rate the reads manage, the
    /// rate needed, and what to do about it. Shown once the read-ahead has
    /// run dry and the reads are under what playback needs.
    fn net_warn(ui: &egui::Ui, rect: Rect, n: media::fetch::NetStats, fps: f64) {
        let can = n.fps();
        if can <= 0.0 || can >= fps * 0.97 || n.ahead > 4 {
            return;
        }
        let text = format!("network: {can:.1} of {fps:.0} fps — frames will drop.  sequence > cache shot locally plays it from this disk");
        let font = FontId::proportional(13.0);
        let p = ui.painter_at(rect);
        let g = p.layout_no_wrap(text, font, Color32::WHITE);
        let at = pos2(rect.center().x - g.size().x / 2.0, rect.max.y - g.size().y - 14.0);
        p.rect_filled(Rect::from_min_size(at, g.size()).expand2(vec2(10.0, 5.0)), 4.0, Color32::from_rgb(170, 30, 30));
        p.galley(at, g, Color32::WHITE);
        ui.ctx().request_repaint();
    }

    fn hud(ui: &egui::Ui, rect: Rect, lines: &[String]) {
        let font = FontId::monospace(HUD_FONT);
        let p = ui.painter_at(rect);
        // Under the key hints (as tall as they wrapped last frame).
        let hints = ui.ctx().data(|d| d.get_temp::<f32>(egui::Id::new(("hint-bottom", rect.min.x as i32)))).unwrap_or(34.0 + HINT_ROW);
        let mut y = rect.min.y + (hints - TITLE_H).max(0.0) + 10.0;
        // A dark plate under the readouts, so they read over any picture.
        let w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as f32 * HUD_FONT * 0.6 + 16.0;
        let h = lines.len() as f32 * (HUD_FONT + 3.0) + 8.0;
        p.rect_filled(Rect::from_min_size(pos2(rect.min.x + 4.0, y - 4.0), vec2(w.min(rect.width() - 8.0), h)), 4.0, Color32::from_black_alpha(150));
        for l in lines {
            let at = pos2(rect.min.x + 10.0, y);
            p.text(at + vec2(1.0, 1.0), Align2::LEFT_TOP, l, font.clone(), Color32::from_black_alpha(200));
            p.text(at, Align2::LEFT_TOP, l, font.clone(), ui.visuals().strong_text_color());
            y += HUD_FONT + 3.0;
        }
    }

    fn source_viewer(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let title = Rect::from_min_size(rect.min, vec2(rect.width(), TITLE_H));
        let pic = Rect::from_min_max(pos2(rect.min.x, title.max.y), pos2(rect.max.x, rect.max.y - SCRUB_H));
        let scrub = Rect::from_min_max(pos2(rect.min.x + 6.0, pic.max.y + 2.0), pos2(rect.max.x - 6.0, rect.max.y));
        let focused = self.focus == Focus::Source;
        let name = self.source.as_ref().map_or("source".to_string(), |c| {
            let n = c.original.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if c.optimised { format!("source: {n}  (optimised copy)") } else { format!("source: {n}") }
        });
        let has_marks = self.source.as_ref().is_some_and(|c| c.marks.inp.is_some() || c.marks.out.is_some());
        let (clear, keys) = self.title_bar(ui, title, &name, focused, has_marks, self.views[0].zoom);
        if clear {
            if let Some(c) = &mut self.source {
                c.marks.clear();
            }
        }
        if keys {
            let on = !self.panes.hints;
            self.panes.set_hints(on);
        }
        let hand = ui.interact(pic, ui.id().with("source-viewer"), Sense::click_and_drag());
        if hand.clicked() {
            self.focus = Focus::Source;
            self.monitor = Focus::Source;
        }
        let navigated = self.views[0].nav(ui, pic, &hand);
        // The marked clip goes onto the timeline by its handle in the title
        // bar, or a SHIFT-drag on the picture (a plain drag scrubs).
        let handle = Rect::from_min_max(pos2(title.max.x - 124.0 - 8.0 - 150.0, title.min.y + 1.0), pos2(title.max.x - 124.0 - 8.0, title.max.y - 1.0));
        let grab = (self.source.is_some() && self.mode == Mode::Edit).then(|| {
            ui.put(handle, egui::Button::new(egui::RichText::new("\u{23f7} drag clip to timeline").size(11.0)).sense(Sense::click_and_drag()))
                .on_hover_text("drag the marked clip onto the timeline (cmd at the drop: insert) — or shift-drag the picture")
        });
        let m = ui.input(|i| i.modifiers) | self.press_mods;
        let shift_drag = !navigated && hand.drag_started_by(egui::PointerButton::Primary) && m.shift && !m.alt && !m.command;
        if let Some(c) = &self.source {
            if grab.as_ref().is_some_and(|g| g.drag_started()) || (shift_drag && pic.contains(ui.input(|i| i.pointer.press_origin()).unwrap_or_default())) {
                let n = c.info.frames as i64;
                let (i, o) = (c.marks.inp.unwrap_or(c.transport.frame() as i64), c.marks.out.unwrap_or(n - 1));
                let fps = Fps::new(c.info.fps_num, c.info.fps_den);
                let n = (o - i + 1).max(1);
                let len = if self.rec.seq.conform { n } else { fps.convert(n, self.rec.seq.fps).max(1) };
                egui::DragAndDrop::set_payload(ui.ctx(), crate::timelineui::Payload::Source { len });
            }
        }
        // A plain drag on the picture scrubs it, as RV does.
        let now = self.source.as_ref().map_or(0, |c| c.transport.frame() as i64);
        if let Some(f) = self.picture_scrub(ui, 0, &hand, navigated, self.source.is_some(), now) {
            if let Some(c) = &mut self.source {
                c.transport.stop();
                c.transport.seek(f.clamp(0, c.info.frames as i64 - 1) as u64);
                c.transport.scrubbing = true;
                self.focus = Focus::Source;
                self.monitor = Focus::Source;
            }
        }
        if let Some((a, _)) = self.trim_frames() {
            // Trim monitor: the outgoing clip's last frame.
            let frame = a.and_then(|a| self.rec.program.layer_frame(&self.rec.seq, a));
            let dims = frame.as_ref().map_or(self.rec.dims, |(_, f)| (f.width, f.height));
            let idt = a.map_or_else(|| Arc::new(crate::viewer::Idt::identity()), |a| self.idt_for(&self.rec.seq.media[a.media].path.clone()));
            Self::picture(ui, pic, 0, dims, frame, a.is_none(), self.views[0], idt, self.view_settings());
            if let Some(c) = &mut self.source {
                c.shown = None;
            }
            Self::hud(ui, pic, &[self.monitor_words().0.to_string(), self.trim_line()]);
            return;
        }
        let src_idt = self.source.as_ref().map(|c| self.idt_for(&c.original.clone()));
        {
            let c = self.source.as_ref();
            let dims = c.map_or((1, 1), |c| (c.info.width, c.info.height));
            let (w, h) = (dims.0.max(1) as f32, dims.1.max(1) as f32);
            let v = self.views[0];
            let scale = (pic.width() / w).min(pic.height() / h) * v.zoom;
            let img = v.apply(pic, Rect::from_center_size(pic.center(), vec2(w, h) * (scale / v.zoom)));
            let mut r = self.pic_rects.get();
            r[0] = Some(img);
            self.pic_rects.set(r);
        }
        let vs = self.view_settings();
        let Some(c) = &mut self.source else {
            let msg = self.error.clone().unwrap_or_else(|| "load a clip from the bin (shift+1, enter)".into());
            ui.painter_at(pic).text(pic.center(), Align2::CENTER_CENTER, msg, FontId::proportional(14.0), ui.visuals().weak_text_color());
            return;
        };
        ui.scope_builder(egui::UiBuilder::new().max_rect(scrub), |ui| scrub_bar(ui, c));
        let target = c.transport.frame();
        let picked = c.server.pick(target, c.shown);
        let exact = picked.as_ref().is_some_and(|(f, _)| *f == target);
        if let Some((f, _)) = &picked {
            if c.shown != Some(*f) {
                self.meters[0].shown();
            }
            c.shown = Some(*f);
        }
        if c.transport.playing() {
            c.transport.account(exact);
        }
        if c.shown.is_some() {
            let frame = picked.map(|(f, fr)| (crate::program::key(&c.server, f), fr));
            Self::picture(ui, pic, 0, (c.info.width, c.info.height), frame, false, self.views[0], src_idt.unwrap_or_else(|| Arc::new(crate::viewer::Idt::identity())), vs);
        } else {
            ui.painter_at(pic).rect_filled(pic, 0.0, Color32::BLACK);
        }
        let st = c.server.stats();
        let f = c.transport.frame();
        let gb = |b: usize| b as f64 / (1u64 << 30) as f64;
        let mut lines = vec![
            format!("{}   {}   {}", c.info.first + f as i64, timecode(f, c.info.fps()), mode(&c.transport)),
            format!("{}  {}x{}  {:.3} fps", c.info.label, c.info.width, c.info.height, c.info.fps()),
            marks_line(c.marks, c.info.first),
            format!("decode {:.1} ms  upload {:.1} ms  {} pictures/s", st.decode_ms, self.upload_ms[0], self.meters[0].rate()),
            format!("cache {:.1}/{:.1} GB  dropped {}", gb(st.cache.bytes), gb(st.cache.budget), c.transport.dropped),
        ];
        lines.extend(st.last_error);
        let net = c.server.source().net().filter(|n| n.network);
        let (playing, fps) = (c.transport.playing(), c.info.fps() * c.transport.speed.abs().max(1.0));
        lines.extend(net.map(net_line));
        lines.extend(self.opt.hud_line());
        lines.extend(self.export.hud_line());
        if self.panes.readouts {
            Self::hud(ui, pic, &lines);
        }
        if let Some(n) = net.filter(|_| playing) {
            Self::net_warn(ui, pic, n, fps);
        }
    }

    fn record_viewer(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let title = Rect::from_min_size(rect.min, vec2(rect.width(), TITLE_H));
        let pic = Rect::from_min_max(pos2(rect.min.x, title.max.y), pos2(rect.max.x, rect.max.y - SCRUB_H));
        let focused = self.focus == Focus::Record;
        let has_marks = self.rec.marks.inp.is_some() || self.rec.marks.out.is_some();
        let (clear, keys) = self.title_bar(ui, title, "record: sequence", focused, has_marks, self.views[1].zoom);
        if clear {
            self.rec.marks.clear();
        }
        if keys {
            let on = !self.panes.hints;
            self.panes.set_hints(on);
        }
        let hand = ui.interact(pic, ui.id().with("record-viewer"), Sense::click_and_drag());
        if hand.clicked() {
            self.focus = Focus::Record;
            self.monitor = Focus::Record;
        }
        let navigated = self.views[1].nav(ui, pic, &hand);
        // A plain drag on the picture scrubs it, as RV does (a drag that
        // starts on a title moves the title instead).
        let on_text = ui.input(|i| i.pointer.press_origin()).is_some_and(|q| self.textv.boxes.iter().any(|b| b.3.contains(q)));
        let now = self.rec.peek.unwrap_or(self.rec.transport.frame() as i64);
        if let Some(f) = self.picture_scrub(ui, 1, &hand, navigated, !on_text, now) {
            let r = &mut self.rec;
            r.transport.stop();
            r.fit_transport();
            r.transport.seek(f.clamp(0, r.seq.duration().max(0)) as u64);
            r.transport.scrubbing = true;
            self.focus = Focus::Record;
            self.monitor = Focus::Record;
        }
        if let Some((_, Some(b))) = self.trim_frames() {
            // Trim monitor: the incoming clip's first frame.
            let frame = self.rec.program.layer_frame(&self.rec.seq, b);
            let dims = frame.as_ref().map_or(self.rec.dims, |(_, f)| (f.width, f.height));
            let idt = self.idt_for(&self.rec.seq.media[b.media].path.clone());
            Self::picture(ui, pic, 1, dims, frame, false, self.views[1], idt, self.view_settings());
            self.rec.shown = None;
            Self::hud(ui, pic, &[self.monitor_words().1.to_string(), self.trim_line()]);
            self.info_strip(ui, rect, pic);
            return;
        }
        let vs = self.view_settings();
        let mut idt = Arc::new(crate::viewer::Idt::identity());
        let r = &mut self.rec;
        let t = r.peek.unwrap_or(r.transport.frame() as i64);
        let mut mix = None;
        let mut medias = (None, None);
        let mut reframe = r.shown_reframe;
        let (frame, black, exact) = match r.program.pick(&r.seq, t, r.shown) {
            Pick::Black => {
                r.shown = Some(t);
                (None, true, true)
            }
            Pick::Frame { key, at, frame, exact, reframe: rf, media, mix: m } => {
                medias = (Some(media), m.as_ref().map(|x| x.4));
                mix = m;
                reframe = rf;
                r.shown_reframe = rf;
                if r.shown != Some(at) {
                    self.meters[1].shown();
                }
                r.shown = Some(at);
                r.dims = (frame.width, frame.height);
                (Some((key, frame)), false, exact)
            }
            Pick::Keep => (None, r.shown.is_none(), false),
        };
        if r.transport.playing() {
            r.transport.account(exact);
        }
        // The sequence raster: the fixed frame every picture is placed in.
        let raster = r.seq.raster.unwrap_or(r.dims);
        let raster_rect = {
            ui.painter_at(pic).rect_filled(pic, 0.0, Color32::from_gray(18));
            let (w, h) = (raster.0.max(1) as f32, raster.1.max(1) as f32);
            let sc = (pic.width() / w).min(pic.height() / h);
            self.views[1].apply(pic, Rect::from_center_size(pic.center(), vec2(w, h) * sc))
        };
        let a = frame.map(|f| (f, reframe));
        let path_of = |m: usize| self.rec.seq.media.get(m).map(|x| x.path.clone()).unwrap_or_default();
        if let Some(m) = medias.0 {
            idt = self.idt_for(&path_of(m));
            self.rec.shown_idt = Some(idt.clone());
        } else if let Some(i) = &self.rec.shown_idt {
            idt = i.clone();
        }
        let mix = mix.map(|(k, f, m, rf, _)| (k, f, m, rf, self.idt_for(&path_of(medias.1.unwrap_or(0)))));
        let r = &self.rec;
        #[cfg(feature = "fold")]
        let ext = self.fold_ext();
        #[cfg(not(feature = "fold"))]
        let ext = None;
        Self::raster_picture(ui, raster_rect, raster, a, r.dims, reframe, black, mix, idt, vs, ext);
        self.text_overlay(ui, raster_rect, raster, t, &hand, navigated);
        let r = &mut self.rec;
        let (decoded, busy, ms) = r.program.stats();
        let lines = vec![
            format!("{}   {}", timecode(t as u64, r.seq.fps.as_f64()), mode(&r.transport)),
            format!(
                "duration {}   {} clips",
                timecode(r.seq.duration() as u64, r.seq.fps.as_f64()),
                r.seq.tracks.iter().map(|t| t.clips.len()).sum::<usize>()
            ),
            marks_line(r.marks, 0),
            format!("decode {ms:.1} ms  upload {:.1} ms  {} pictures/s", self.upload_ms[1], self.meters[1].rate()),
            format!("decoded {decoded}  busy {busy}  dropped {}", r.transport.dropped),
        ];
        let mut lines = lines;
        lines.extend(self.reframe_line());
        let net = self.rec.program.net();
        let (playing, fps) = (self.rec.transport.playing(), self.rec.seq.fps.as_f64() * self.rec.transport.speed.abs().max(1.0));
        lines.extend(net.map(net_line));
        lines.extend(self.local_caching_line());
        lines.extend(self.export.hud_line());
        if let Some(p) = &self.post {
            lines.insert(0, format!("post: {}", p.title));
        }
        if self.panes.readouts {
            Self::hud(ui, pic, &lines);
        }
        if let Some(n) = net.filter(|_| playing) {
            Self::net_warn(ui, pic, n, fps);
        }
        self.info_strip(ui, rect, pic);
    }

    /// Under the record picture: targets, trim mode, undo depth, or the
    /// last notice.
    fn info_strip(&self, ui: &egui::Ui, rect: Rect, pic: Rect) {
        let r = &self.rec;
        let info = Rect::from_min_max(pos2(rect.min.x, pic.max.y), rect.max);
        let tg = &r.targets;
        let v = tg.video.map_or("none".to_string(), |i| r.seq.tracks[i].name.to_string());
        let text = match &self.notice {
            // Messages show in the timeline panel when it is there.
            Some((n, at)) if at.elapsed().as_secs() < 4 && !self.panes.shown(crate::panes::Pane::Timeline) => n.clone(),
            _ => format!(
                "{} tool    trims {} (B)    targets  video {v}  audio {}    undo {} / redo {}",
                r.tool.name(),
                if r.ripple { "RIPPLE" } else { "plain" },
                tg.audio.map_or("off", |a| r.seq.tracks[a].name),
                r.history.depth().0,
                r.history.depth().1
            ),
        };
        ui.painter_at(info).text(info.left_center() + vec2(8.0, 0.0), Align2::LEFT_CENTER, text, FontId::proportional(12.0), ui.visuals().text_color());
    }

    /// The selected edit, in words.
    fn trim_line(&self) -> String {
        match self.rec.sel {
            crate::timelineui::Selection::Edit { track, at, side } => format!(
                "{} cut at {}  {:?}  {}",
                self.rec.seq.tracks[track].name,
                timecode(at.max(0) as u64, self.rec.seq.fps.as_f64()),
                side,
                if self.rec.ripple && side != model::Side::Roll { "ripple" } else { "" }
            ),
            _ => String::new(),
        }
    }

    fn timeline(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let r = &mut self.rec;
        if r.transport.playing() {
            r.view.follow(r.transport.frame() as i64);
        }
        let focused = self.focus == Focus::Record;
        // A slip (by key, or a drag that did not set the band itself) shows
        // the clip's handles too.
        let mut overlay = self.overlay.clone();
        if let Some(b) = self.slip_band().filter(|_| overlay.handles.is_empty()) {
            overlay.handles.push(b);
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
        }
        let r = &mut self.rec;
        let ev = ui
            .scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                r.view.ui(ui, &r.seq, r.transport.frame() as i64, r.marks, r.targets, focused, &r.sel, r.ripple, &overlay, r.tool, self.hold.map(|h| h.label()))
            })
            .inner;
        // Messages about what was just done show here, by the tracks, on a
        // plate at the timeline's top right (the record strip keeps the
        // status line).
        if let Some((n, at)) = &self.notice {
            let age = at.elapsed().as_secs_f32();
            if age < 4.0 {
                let fade = (4.0 - age).min(0.5) / 0.5;
                let p = ui.painter_at(rect);
                let col = Color32::from_white_alpha((235.0 * fade) as u8);
                let g = p.layout_no_wrap(n.clone(), FontId::proportional(12.0), col);
                let br = Rect::from_min_size(pos2(rect.max.x - g.size().x - 18.0, rect.min.y + 22.0), g.size()).expand2(vec2(6.0, 3.0));
                p.rect_filled(br, 4.0, Color32::from_black_alpha((210.0 * fade) as u8));
                p.galley(br.min + vec2(6.0, 3.0), g, col);
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
        if self.timeline_mouse(&ev) {
            return;
        }
        if let timelineui::Event::Track(t, op) = ev {
            self.track_op(t, op);
            return;
        }
        if let timelineui::Event::ClearMarks = ev {
            self.rec.marks.clear();
            self.say("record in and out cleared");
            return;
        }
        let r = &mut self.rec;
        let down = ui.input(|i| i.pointer.primary_down());
        let mut scrubbed = false;
        match ev {
            timelineui::Event::Scrub(f) => {
                r.transport.stop();
                r.fit_transport();
                r.transport.seek(f as u64);
                scrubbed = true;
                self.focus = Focus::Record;
                self.monitor = Focus::Record;
            }
            timelineui::Event::Targets(t) => r.targets = t,
            _ => {}
        }
        r.transport.scrubbing = down && (scrubbed || r.transport.scrubbing);
    }

    fn menubar(&mut self, ui: &mut egui::Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("file", |ui| {
                if ui.button("open project…  (cmd O)").clicked() {
                    self.open_project_dialog();
                }
                ui.menu_button("open recent", |ui| {
                    if self.project.recent.is_empty() {
                        ui.weak("none yet");
                    }
                    for p in self.project.recent.clone() {
                        if ui.button(p.display().to_string()).clicked() {
                            self.open_project(&p);
                        }
                    }
                });
                if ui.button("save  (cmd S)").clicked() {
                    self.save(false);
                }
                if ui.button("save as…  (cmd shift S)").clicked() {
                    self.save(true);
                }
                if ui.button("open marketing post…").clicked() {
                    self.open_posts_window();
                }
                ui.separator();
                if ui.button("render…  (cmd M)").clicked() {
                    self.ask_render();
                }
                if ui.add_enabled(self.export.busy(), egui::Button::new("cancel render")).clicked() {
                    self.export.cancel();
                }
                ui.separator();
                if ui.button("import subtitles (SRT)…").on_hover_text("text clips on the targeted video track, in the subtitle style").clicked() {
                    self.srt_dialog(false);
                }
                if ui.button("export subtitles (SRT)…").on_hover_text("the text clips of the video track with the most").clicked() {
                    self.srt_dialog(true);
                }
                if ui.button("import media…  (cmd I)").clicked() {
                    self.import();
                }
                if ui.add_enabled(!self.project.missing.is_empty(), egui::Button::new("relink media…")).clicked() {
                    self.relink_open = true;
                }
                if ui.button("sequence settings…").clicked() {
                    let s = &self.rec.seq;
                    let (w, h) = s.raster.unwrap_or((1920, 1080));
                    self.seq_settings = Some((w, h, s.fps.num, s.fps.den));
                }
                let can = self.current().is_some_and(|(_, opt, intra)| !opt && !intra);
                if ui.add_enabled(can, egui::Button::new("optimise source…  (cmd T)")).clicked() {
                    self.opt.ask();
                }
                if ui.button("media caches…  (optimised, network copies)").clicked() {
                    self.opt.show_cache = true;
                }
                ui.separator();
                if ui.button("quit  (cmd Q)").clicked() && self.request_quit() {
                    self.quit = true;
                }
            });
            ui.menu_button("view", |ui| {
                for p in crate::panes::ALL {
                    let mut on = self.panes.open.contains(&p);
                    if ui.checkbox(&mut on, format!("{}  (W over it closes it)", p.name())).changed() {
                        self.set_pane_open(p, on);
                    }
                }
                if ui.button("show every panel  (shift W)").clicked() {
                    self.show_all_panes();
                }
                ui.separator();
                let mut hints = self.panes.hints;
                if ui.checkbox(&mut hints, "key hints over the viewers  (H)").changed() {
                    self.panes.set_hints(hints);
                }
                let mut ro = self.panes.readouts;
                if ui.checkbox(&mut ro, "viewer readouts: frame, decode, cache  (shift H)").changed() {
                    self.panes.set_readouts(ro);
                }
                ui.separator();
                if ui.button("fit both viewers  (R over a viewer: that one)").clicked() {
                    self.views = Default::default();
                }
                if ui.button("viewers the same size  (shift R over a viewer)").clicked() {
                    self.views[0] = self.views[1];
                    self.match_viewers();
                }
                ui.weak("Q over a panel: just that one (Q / Esc: back)");
                ui.weak("drag a panel by its grip to move it; out of the window: a window of its own");
            });
            ui.menu_button("colour", |ui| self.colour_menu(ui));
            ui.menu_button("window", |ui| {
                if ui.button("reset layout").on_hover_text("every panel back in the main window, as at first").clicked() {
                    self.dock_reset = true;
                }
                if ui.button("show every panel  (shift W)").clicked() {
                    self.show_all_panes();
                }
            });
            #[cfg(feature = "fold")]
            ui.menu_button("wob", |ui| {
                if ui.button("make the selected clip a wob, or dive into it  (cmd shift W / Enter)").clicked() {
                    self.fold_make();
                }
                if ui.button("back to footage  (cmd alt W)").clicked() {
                    self.fold_clear();
                }
                if ui.button("save the open wob").clicked() {
                    self.fold_save();
                }
                match self.fold_where() {
                    1 => {
                        if ui.button("back to scrub  (cmd shift W)").clicked() {
                            self.fold_close_pane();
                        }
                    }
                    0 => {
                        if ui.button("open wobble, whole window").clicked() {
                            self.fold_open_pane();
                        }
                    }
                    _ => {}
                }
                if self.fold_where() != 2 && ui.button("open wobble in its own window").clicked() {
                    self.fold_own_window();
                }
                let shown = self.panes.open.contains(&crate::panes::Pane::Controls);
                if ui.button(if shown { "hide wob controls" } else { "show wob controls" }).clicked() {
                    if shown {
                        self.set_pane_open(crate::panes::Pane::Controls, false);
                    } else {
                        self.fold_open_controls();
                    }
                }
            });
            ui.menu_button("sequence", |ui| {
                if ui.button("cache shot locally").on_hover_text("copy every frame of the image sequences in the source and the timeline that sit on network storage to this disk, so they play at disk speed").clicked() {
                    self.cache_locally();
                }
                if ui.button("new text  (cmd shift T)").on_hover_text("a title at the playhead, above the targeted track").clicked() {
                    self.new_text(None, None, "Title");
                }
                let sel = self.selected_text();
                if ui.add_enabled(sel.is_some(), egui::Button::new("edit text…  (Enter, double-click)")).clicked() {
                    if let Some((t, i)) = sel {
                        let st = self.rec.seq.tracks[t].clips[i].start;
                        self.open_text(t, st);
                    }
                }
                ui.separator();
                if ui.button("add dissolve  (cmd D)").on_hover_text("on the selected edit point, between two selected clips, or the cut nearest the playhead").clicked() {
                    self.add_dissolve(None);
                }
                if ui.button("remove dissolve  (cmd shift D)").clicked() {
                    self.remove_dissolve();
                }
                ui.separator();
                if ui.button("cut at playhead  (cmd K)").clicked() {
                    self.razor_playhead(false);
                }
                if ui.button("cut every track at playhead  (cmd shift K)").clicked() {
                    self.razor_playhead(true);
                }
                if ui.button("fit to fill  (shift .)").clicked() {
                    self.fit_to_fill();
                }
                if ui.button("clip speed…  (cmd R)").clicked() {
                    self.ask_speed();
                }
                if ui.button("conform clip on / off  (alt R)").on_hover_text("one source frame per sequence frame: a clip at another rate plays every frame, no drops or repeats (its length changes; its sound is muted)").clicked() {
                    self.toggle_conform();
                }
                #[cfg(feature = "wobble")]
                {
                    ui.separator();
                    ui.menu_button("experimental", |ui| {
                        let mut on = self.panes.experimental_wobble;
                        if ui.checkbox(&mut on, "Wobble clips  (cmd shift W on a clip)").changed() {
                            self.panes.set_experimental_wobble(on);
                        }
                        if ui.add_enabled(on, egui::Button::new("open clip in Wobble  (cmd shift W)")).clicked() {
                            self.wobble_open();
                        }
                    });
                }
                ui.separator();
                ui.weak("tools");
                for t in crate::timelineui::Tool::ALL {
                    let k = if t.key().is_empty() { "hold shift A / D" } else { t.key() };
                    if ui.radio(self.rec.tool == t, format!("{}  ({k})", t.name())).clicked() {
                        self.set_tool(t);
                    }
                }
                ui.separator();
                let mut rip = self.rec.ripple;
                if ui.checkbox(&mut rip, "ripple trims  (B)").changed() {
                    self.rec.ripple = rip;
                }
                let mut snap = self.rec.snap;
                if ui.checkbox(&mut snap, "snapping  (T)").changed() {
                    self.rec.snap = snap;
                }
            });
            ui.menu_button("help", |ui| {
                if ui.button("hotkey tour").clicked() {
                    self.tour.replay();
                    ui.close();
                }
                if ui.button("request a feature…").clicked() {
                    self.feedback.request();
                    ui.close();
                }
            });
            ui.menu_button("edit", |ui| {
                if ui.button("undo  (cmd Z)").clicked() {
                    self.undo(false);
                }
                if ui.button("redo  (cmd shift Z)").clicked() {
                    self.undo(true);
                }
                ui.separator();
                if ui.button("insert  (,)").clicked() {
                    self.edit(true);
                }
                if ui.button("overwrite  (.)").clicked() {
                    self.edit(false);
                }
                if ui.button("append at the end  (M)").clicked() {
                    self.append();
                }
            });
            // The screen recorder (wobble's): record / stop, a screenshot.
            ui.separator();
            let rec = self.recorder.lock().map(|r| r.bar_button(ui, "cmd shift R")).unwrap_or(false);
            if rec {
                self.toggle_recording(ui.ctx());
            }
            let shot = self.shot.lock().map(|s| s.bar_button(ui)).unwrap_or(false);
            if shot {
                if let Ok(mut s) = self.shot.lock() {
                    s.request();
                }
                ui.ctx().request_repaint();
            }
            let err = self.recorder.lock().ok().and_then(|r| r.error.clone());
            if let Some(e) = err {
                ui.colored_label(Color32::from_rgb(230, 80, 70), e);
            }
            let note = self.shot.lock().ok().and_then(|s| s.fresh_note().cloned());
            match note {
                Some(Ok(path)) => {
                    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if ui.link(format!("saved {name}")).on_hover_text(format!("{}\nclick to show it", path.display())).clicked() {
                        crate::recorder::share::reveal(&path);
                    }
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(500));
                }
                Some(Err(e)) => {
                    ui.colored_label(Color32::from_rgb(230, 80, 70), e);
                }
                None => {}
            }
            // The mode switch (` toggles).
            ui.separator();
            for (m, label) in [(Mode::Edit, "edit"), (Mode::Selects, "selects")] {
                if ui.selectable_label(self.mode == m, label).on_hover_text("` (backtick) switches edit / selects").clicked() && self.mode != m {
                    self.toggle_mode();
                }
            }
            // Feedback, in red at the far right (as wobble has it).
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| self.feedback.button(ui));
        });
    }

    /// cmd shift R / the record button: start a take of the window (or
    /// the record viewer, per the recorder's options), or stop it.
    pub(crate) fn toggle_recording(&mut self, ctx: &egui::Context) {
        let Some(device) = self.device.clone() else { return };
        let ppp = ctx.pixels_per_point();
        let screen = ctx.content_rect();
        let (sw, sh) = ((screen.width() * ppp).round() as u32, (screen.height() * ppp).round() as u32);
        let viewer = self.pane_rect(crate::panes::Pane::Record);
        let Ok(mut r) = self.recorder.lock() else { return };
        if r.recording() {
            r.stop(&device);
            return;
        }
        let rect = match (r.opts.area, viewer) {
            (crate::recorder::Area::Viewer, Some(v)) => [v.min.x * ppp, v.min.y * ppp, v.width() * ppp, v.height() * ppp],
            _ => [0.0, 0.0, sw as f32, sh as f32],
        };
        r.start(&device, rect, sw, sh, ppp);
    }

    /// Sequence settings: the record raster (always) and the rate (while
    /// the sequence is empty: frames are its unit of time).
    fn settings_window(&mut self, ctx: &egui::Context) {
        let Some((mut w, mut h, mut num, mut den)) = self.seq_settings else { return };
        let mut open = true;
        let mut apply = false;
        let empty = self.rec.seq.is_empty();
        let mut set_frame: Option<&str> = None;
        let mut conform = self.rec.seq.conform;
        egui::Window::new("sequence settings").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("raster");
                ui.add(egui::DragValue::new(&mut w).range(16..=16384));
                ui.label("x");
                ui.add(egui::DragValue::new(&mut h).range(16..=16384));
                for (pw, ph) in [(1920u32, 1080u32), (3840, 2160), (1080, 1920), (1080, 1080)] {
                    if ui.small_button(format!("{pw}x{ph}")).clicked() {
                        (w, h) = (pw, ph);
                    }
                }
            });
            ui.add_enabled_ui(empty, |ui| {
                ui.horizontal(|ui| {
                    ui.label("frame rate");
                    ui.add(egui::DragValue::new(&mut num).range(1..=240_000));
                    ui.label("/");
                    ui.add(egui::DragValue::new(&mut den).range(1..=1001));
                    for (n, d) in [(24, 1), (24000, 1001), (25, 1), (30, 1), (30000, 1001), (60, 1)] {
                        if ui.small_button(format!("{:.3}", n as f64 / d as f64)).clicked() {
                            (num, den) = (n, d);
                        }
                    }
                });
            });
            if !empty {
                ui.weak("the rate is fixed once the sequence has clips");
            }
            ui.checkbox(&mut conform, "conform clips at other rates (new edits)").on_hover_text(
                "a clip edited in at another frame rate plays one source frame per sequence frame: no drops or repeats, its length and speed change, its sound is muted. alt R toggles one clip.",
            );
            if let Some(p) = &self.post {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(format!("post frame shape ({})", if p.frame.is_empty() { "original" } else { &p.frame }));
                    for (shape, _) in crate::marketing::SHAPES {
                        if ui.small_button(shape).on_hover_text("sets the raster and writes the post's frame shape to its post.json").clicked() {
                            set_frame = Some(shape);
                        }
                    }
                });
            }
            apply = ui.button("apply").clicked();
        });
        self.seq_settings = Some((w, h, num, den));
        if conform != self.rec.seq.conform {
            self.rec.seq.conform = conform;
            self.changed(false);
            self.say(if conform { "new edits at other rates conform" } else { "new edits play at real time" });
        }
        if let Some(f) = set_frame {
            if let Err(e) = self.set_post_frame(f) {
                self.say(e);
            }
            if let Some((rw, rh)) = self.rec.seq.raster {
                self.seq_settings = Some((rw, rh, num, den));
            }
            return;
        }
        if apply {
            self.rec.seq.raster = Some((w, h));
            if empty {
                self.rec.seq.fps = Fps::new(num, den);
            }
            self.changed(false);
            open = false;
        }
        if !open {
            self.seq_settings = None;
        }
    }

    /// cmd+R: the clip speed as a multiplier (negative plays backwards).
    fn speed_window(&mut self, ctx: &egui::Context) {
        let Some(mut v) = self.speed_dialog else { return };
        let mut open = true;
        let mut apply = false;
        egui::Window::new("clip speed").open(&mut open).collapsible(false).resizable(false).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("speed");
                let r = ui.add(egui::DragValue::new(&mut v).speed(0.01).range(-16.0..=16.0).suffix("x"));
                if self.speed_dialog_focus {
                    r.request_focus();
                    self.speed_dialog_focus = false;
                }
                for p in [0.5, 1.0, 2.0, -1.0] {
                    if ui.small_button(format!("{p}x")).clicked() {
                        v = p;
                    }
                }
            });
            ui.weak("negative plays backwards; nearest frame, no blending; audio is muted off 1x");
            apply = ui.button("apply  (enter)").clicked() || ui.input(|i| i.key_pressed(Key::Enter));
        });
        if ctx.input(|i| i.key_pressed(Key::Escape)) {
            open = false;
        }
        self.speed_dialog = Some(v);
        if apply {
            self.apply_speed(v);
            open = false;
        }
        if !open {
            self.speed_dialog = None;
        }
    }

}

impl imfram::shell::UiApp for Editor {
    fn ui(&mut self, ui: &mut egui::Ui) {
        imfram::theme::sync(ui.ctx());
        let now = Instant::now();
        let dt = now.duration_since(self.last_tick).as_secs_f64().min(0.25);
        self.last_tick = now;
        // Scripting calls first: their edits are drawn this pass.
        self.api_drain();
        if let Some(m) = ui.input(|i| i.events.iter().rev().find_map(|e| if let egui::Event::PointerButton { pressed: true, modifiers, .. } = e { Some(*modifiers) } else { None })) {
            self.press_mods = m;
        }
        #[cfg(feature = "fold")]
        {
            self.fold_begin();
            // cmd shift W is Scrub's even over wobble: in and out of it.
            if take(ui, Modifiers::COMMAND | Modifiers::SHIFT, Key::W, false) {
                if self.fold_where() == 1 {
                    self.fold_close_pane();
                } else {
                    self.fold_make();
                }
            }
            match std::mem::take(&mut self.fold.ask) {
                1 => self.fold_close_pane(),
                2 => self.fold_own_window(),
                _ => {}
            }
            self.fold_layout();
        }
        #[cfg(feature = "fold")]
        let wob_keys = self.fold_has_keys(ui);
        #[cfg(not(feature = "fold"))]
        let wob_keys = false;
        // The tour's card and the feature form keep the keys to themselves.
        if self.tour.open || self.feedback.up() {
            self.hold = None;
        } else if !wob_keys {
            self.keys(ui);
        }
        // The record playhead may sit one past the last frame (where the
        // next edit goes), so the record side has duration + 1 positions.
        self.rec.fit_transport();
        self.preroll();
        let by_sound = self.sync_audio();
        if let Some(c) = &mut self.source {
            if by_sound != Some(Focus::Source) {
                c.transport.tick(dt);
            }
        }
        if by_sound != Some(Focus::Record) {
            self.rec.transport.tick(dt);
        }

        let full = ui.max_rect();
        let menu = Rect::from_min_max(full.min, pos2(full.max.x, full.min.y + MENU_H));
        ui.scope_builder(egui::UiBuilder::new().max_rect(menu), |ui| self.menubar(ui));
        let body = Rect::from_min_max(pos2(full.min.x, menu.max.y), full.max);
        self.feed();
        // The panels: the dock in the room under the menubar.
        self.panes.begin_host(crate::panes::MAIN);
        match self.docks.take() {
            // Selects mode lays the main window out itself (the dock's
            // layout is left as it was, for edit mode).
            Some(d) if self.mode == Mode::Selects => {
                ui.scope_builder(egui::UiBuilder::new().max_rect(body), |ui| self.selects_body(ui, body));
                self.docks = Some(d);
            }
            None if self.mode == Mode::Selects => {
                ui.scope_builder(egui::UiBuilder::new().max_rect(body), |ui| self.selects_body(ui, body));
            }
            Some(mut d) => {
                ui.scope_builder(egui::UiBuilder::new().max_rect(body), |ui| d.draw_external(crate::panes::MAIN, ui, self));
                let released = ui.input(|i| i.pointer.any_released());
                self.dock_sync(&d);
                self.panes.note_layout(&d, released);
                self.docks = Some(d);
            }
            None => {
                // No window attached (an editor made for a test): the
                // default layout, drawn without the dock's machinery.
                let t = self.panes.default_tree();
                let lay = t.layout(body, 6.0);
                for (leaf, r) in lay.leaves {
                    if let Some(p) = t.active(leaf) {
                        ui.scope_builder(egui::UiBuilder::new().max_rect(r), |ui| self.tab_body(ui, p));
                    }
                }
            }
        }
        self.panes.end_host(crate::panes::MAIN);
        self.feed();
        #[cfg(feature = "fold")]
        self.fold_end(ui.ctx());

        // What is being dragged, by the pointer.
        if let Some(pl) = egui::DragAndDrop::payload::<crate::timelineui::Payload>(ui.ctx()) {
            if let Some(q) = ui.input(|i| i.pointer.hover_pos()) {
                let what = match &*pl {
                    crate::timelineui::Payload::Source { len } => format!("source clip, {len} frames"),
                    crate::timelineui::Payload::File(p) => p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                    crate::timelineui::Payload::Files(v) => format!("{} clips, laid down in order", v.len()),
                    crate::timelineui::Payload::Clips(v) if v.len() == 1 => "subclip".into(),
                    crate::timelineui::Payload::Clips(v) => format!("{} clips, laid down in order", v.len()),
                    crate::timelineui::Payload::Dissolve => "cross dissolve: onto a cut".into(),
                    crate::timelineui::Payload::Text => "text: onto a video track".into(),
                };
                let painter = ui.ctx().layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("drag-label")));
                let g = painter.layout_no_wrap(what, FontId::proportional(12.0), Color32::WHITE);
                let r = Rect::from_min_size(q + vec2(14.0, 10.0), g.size()).expand(4.0);
                painter.rect_filled(r, 3.0, Color32::from_black_alpha(200));
                painter.galley(r.min + vec2(4.0, 4.0), g, Color32::WHITE);
            }
        }
        if self.rec.history.settled() {
            self.rec.history.commit(&self.rec.seq);
        }
        self.project_tick();
        if self.project_windows(ui.ctx()) {
            self.quit = true;
        }
        self.settings_window(ui.ctx());
        self.feedback.windows(ui.ctx());
        if self.tour.show(ui.ctx()) && !self.panes.tour_seen {
            self.panes.set_tour_seen();
        }
        self.export_windows(ui.ctx());
        self.posts_ui(ui.ctx());
        self.speed_window(ui.ctx());
        self.text_window(ui.ctx());
        self.recorder_tick(ui.ctx());
        #[cfg(feature = "clips")]
        self.clips_window(ui.ctx());
        #[cfg(feature = "wobble")]
        {
            self.wobble_tick();
            self.wobble_window(ui.ctx());
            if self.wobble_busy() {
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
            }
        }
        let cur = self.current().map(|(p, o, i)| (p.to_path_buf(), o, i));
        if self.opt.windows(ui.ctx(), cur.as_ref().map(|(p, o, i)| (p.as_path(), *o, *i))) {
            self.reopen();
        }
        if let Some(done) = self.opt.poll() {
            let in_seq = self.rec.seq.media.iter().any(|m| m.path == done);
            if self.source.as_ref().is_some_and(|c| c.original == done) || in_seq {
                self.reopen();
            }
        }
        if self.opt.busy() || self.export.busy() || !self.local_caching.is_empty() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
        let busy = self.source.as_ref().is_some_and(|c| c.transport.playing() || c.server.stats().in_flight + c.server.stats().queued > 0)
            || self.rec.transport.playing()
            || self.rec.program.stats().1 > 0;
        if busy {
            ui.ctx().request_repaint();
        }
    }
}

/// The input-space choices for `path`: by its format, the working space,
/// or a named colourspace. Returns the pick (`Some(None)`: by format).
pub(crate) fn input_space_menu(ui: &mut egui::Ui, c: &model::ColourSettings, path: &Path) -> Option<Option<colour::color::InputSpace>> {
    use colour::color::{Colorspace, InputSpace};
    let now = c.inputs.get(path).copied();
    let mut out = None;
    let by_format = model::ColourSettings::default_for(path).unwrap_or(c.working).label();
    if ui.radio(now.is_none(), format!("by format ({by_format})")).clicked() {
        out = Some(None);
    }
    if ui.radio(now == Some(InputSpace::Working), format!("the working space ({})", c.working.label())).clicked() {
        out = Some(Some(InputSpace::Working));
    }
    ui.separator();
    for (name, cs) in Colorspace::PRESETS {
        if ui.radio(now == Some(InputSpace::Explicit(cs)), name).clicked() {
            out = Some(Some(InputSpace::Explicit(cs)));
        }
    }
    out
}

/// The read-ahead of sequences on network storage, in a readout line.
fn net_line(n: media::fetch::NetStats) -> String {
    format!(
        "network {:.0} MB/s ({:.1} fps)  {} ahead  local copies {} / {}",
        n.bytes_per_sec / 1e6,
        n.fps(),
        n.ahead,
        n.local_hits,
        n.local_hits + n.remote_reads
    )
}

fn play_state(t: &Transport) -> PlayState {
    PlayState {
        frame: t.frame(),
        dir: t.dir(),
        step: t.step(DISPLAY_HZ),
        playing: t.playing(),
        scrubbing: t.scrubbing,
        looping: t.looping,
    }
}

fn mode(t: &Transport) -> String {
    if t.playing() {
        format!("{}{}x", if t.speed < 0.0 { "rev " } else { "" }, t.speed.abs())
    } else {
        "stopped".into()
    }
}

fn marks_line(m: Marks, first: i64) -> String {
    let f = |x: Option<i64>| x.map_or("-".into(), |v| (v + first).to_string());
    let dur = match (m.inp, m.out) {
        (Some(i), Some(o)) => format!("  dur {}", o - i + 1),
        _ => String::new(),
    };
    format!("in {}  out {}{dur}", f(m.inp), f(m.out))
}

/// The source's scrub bar, with its cached frames along the bottom edge
/// and its in/out marks.
fn scrub_bar(ui: &mut egui::Ui, c: &mut SourceClip) {
    let first = c.scrub.range.0;
    c.scrub.playhead = first + c.transport.frame() as i64;
    c.scrub.playing = c.transport.playing();
    c.scrub.in_out = match (c.marks.inp, c.marks.out) {
        (None, None) => None,
        (i, o) => Some((first + i.unwrap_or(0), first + o.unwrap_or(c.info.frames as i64 - 1))),
    };
    let events = c.scrub.show(ui, &[]);
    let mut scrubbed = false;
    for e in events {
        match e {
            timeline::Event::Scrub(f) => {
                c.transport.stop();
                c.transport.seek((f - first).max(0) as u64);
                scrubbed = true;
            }
            timeline::Event::Play | timeline::Event::Pause => c.transport.toggle(),
            _ => {}
        }
    }
    let down = ui.input(|i| i.pointer.primary_down());
    c.transport.scrubbing = down && (scrubbed || c.transport.scrubbing);

    let strip = c.scrub.strip_rect();
    let bar = Rect::from_min_max(pos2(strip.min.x, strip.max.y - CACHE_BAR_H), strip.max);
    let p = ui.painter_at(strip);
    let fill = imfram::theme::ok();
    let mut cached = c.server.cached_frames();
    cached.sort_unstable();
    let mut i = 0;
    while i < cached.len() {
        let mut j = i;
        while j + 1 < cached.len() && cached[j + 1] == cached[j] + 1 {
            j += 1;
        }
        let x0 = timeline::frame_to_px(c.scrub.view, strip.min.x, (first + cached[i] as i64) as f64 - 0.5);
        let x1 = timeline::frame_to_px(c.scrub.view, strip.min.x, (first + cached[j] as i64) as f64 + 0.5);
        p.rect_filled(Rect::from_x_y_ranges(x0..=x1.max(x0 + 1.0), bar.y_range()), 0.0, fill);
        i = j + 1;
    }
}

/// Take a key press out of this frame's input (so no widget sees it too).
/// Modifiers must match exactly: I and shift+I are different keys here.
pub fn take(ui: &mut egui::Ui, mods: Modifiers, key: Key, repeats: bool) -> bool {
    ui.input_mut(|i| {
        let before = i.events.len();
        i.events.retain(|e| {
            !matches!(e, egui::Event::Key { key: k, pressed: true, repeat, modifiers, .. }
                if *k == key && (repeats || !*repeat) && modifiers.matches_exact(mods))
        });
        i.events.len() != before
    })
}

pub fn timecode(frame: u64, fps: f64) -> String {
    let base = fps.round().max(1.0) as u64;
    let (s, f) = (frame / base, frame % base);
    format!("{:02}:{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60, f)
}

#[cfg(test)]
mod tests {
    #[test]
    fn timecode_counts_whole_frames_at_the_rounded_rate() {
        assert_eq!(super::timecode(0, 24.0), "00:00:00:00");
        assert_eq!(super::timecode(24 * 61 + 5, 24.0), "00:01:01:05");
        assert_eq!(super::timecode(30, 29.97), "00:00:01:00");
    }
}

// --- the panels on the dock (`panes`) --------------------------------------------------

impl Editor {
    /// sequence > cache shot locally: every image sequence on network
    /// storage in the source and the timeline, copied frame by frame to
    /// the local disk cache in the background.
    pub(crate) fn cache_locally(&mut self) {
        let mut paths: Vec<PathBuf> = self.rec.seq.media.iter().filter(|m| !m.generated).map(|m| m.path.clone()).collect();
        if let Some(c) = &self.source {
            paths.push(c.original.clone());
        }
        paths.sort();
        paths.dedup();
        let mut n = 0;
        for p in paths {
            if !media::sequence::is_image(&p) || !media::netio::is_network(&p) || self.local_caching.iter().any(|(name, ..)| *name == p.display().to_string()) {
                continue;
            }
            let Some((pat, _)) = media::sequence::Pattern::of(&p) else { continue };
            let Ok((a, b)) = pat.scan() else { continue };
            let frames = (b - a + 1).max(0) as u64;
            let done = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            media::fetch::cache_locally(frames, move |i| pat.path(a + i as i64), done.clone(), stop.clone());
            self.local_caching.push((p.display().to_string(), done, frames, stop));
            n += 1;
        }
        self.say(if n == 0 { "no image sequences on network storage to copy".to_string() } else { format!("copying {n} sequence(s) to this disk") });
    }

    /// "cache shot locally" in progress, in words (and the finished ones
    /// dropped).
    fn local_caching_line(&mut self) -> Option<String> {
        use std::sync::atomic::Ordering;
        self.local_caching.retain(|(_, d, n, _)| d.load(Ordering::Relaxed) < *n);
        let (d, n) = self.local_caching.iter().fold((0, 0), |(a, b), (_, d, n, _)| (a + d.load(Ordering::Relaxed), b + n));
        (n > 0).then(|| format!("copying to this disk: {d} of {n} frames"))
    }

    /// The panel the keys go to now: the one under the pointer (in the
    /// window being drawn), else the focused one. Marks (I / O) and the
    /// pane keys follow it.
    pub(crate) fn key_target(&self, ui: &egui::Ui) -> Option<crate::panes::Pane> {
        use crate::panes::Pane;
        let hover = ui.input(|i| i.pointer.hover_pos()).and_then(|q| self.panes.at(q)).map(|(p, _)| p);
        hover.or(match self.focus {
            Focus::Bin => Some(Pane::Bin),
            Focus::Source => Some(Pane::Source),
            Focus::Record => Some(Pane::Record),
        })
    }

    /// One panel's body, filling `ui`.
    pub(crate) fn tab_body(&mut self, ui: &mut egui::Ui, pane: crate::panes::Pane) {
        let rect = ui.max_rect();
        self.panes.drew(pane, rect);
        self.tab_contents(ui, pane, rect);
        // The panel the keys go to wears a thin accent edge.
        if self.key_target(ui) == Some(pane) {
            let c = ui.visuals().selection.stroke.color.gamma_multiply(0.55);
            ui.painter().rect_stroke(rect.shrink(0.5), 2.0, Stroke::new(1.0, c), egui::StrokeKind::Inside);
            self.key_panel = Some(pane);
        }
    }

    fn tab_contents(&mut self, ui: &mut egui::Ui, pane: crate::panes::Pane, rect: Rect) {
        use crate::panes::Pane;
        match pane {
            Pane::Bin => {
                #[cfg(feature = "clips")]
                self.bin_drop(ui, rect);
                let bin_focused = self.focus == Focus::Bin;
                let act = ui
                    .scope_builder(egui::UiBuilder::new().max_rect(rect.shrink(4.0)), |ui| {
                        if ui.interact(rect, ui.id().with("bin-panel"), Sense::click()).clicked() {
                            self.focus = Focus::Bin;
                        }
                        if !self.bin.prefs_in {
                            (self.bin.tiles, self.bin.tile, self.bin.prefs_in) = (self.panes.bin_tiles, self.panes.bin_tile, true);
                        }
                        self.bin.ui(ui, bin_focused)
                    })
                    .inner;
                if (self.bin.tiles, self.bin.tile) != (self.panes.bin_tiles, self.panes.bin_tile) {
                    (self.panes.bin_tiles, self.panes.bin_tile) = (self.bin.tiles, self.bin.tile);
                    self.panes.changed();
                }
                if bin_focused {
                    ui.painter().rect_stroke(rect.shrink(1.0), 0.0, Stroke::new(1.0, ui.visuals().selection.bg_fill), egui::StrokeKind::Inside);
                }
                if let bin::Action::Browse = act {
                    self.import();
                }
                if let bin::Action::Load(p, r) = act {
                    self.load_item(&p, r);
                }
            }
            Pane::Timeline => {
                // A bar like the viewers': the dock's grip at its left.
                let bar = Rect::from_min_size(rect.min, vec2(rect.width(), TITLE_H));
                let focused = self.focus == Focus::Record;
                let v = ui.visuals().clone();
                ui.painter().rect_filled(bar, 0.0, v.widgets.noninteractive.bg_fill);
                let gw = ui
                    .scope_builder(egui::UiBuilder::new().max_rect(bar.shrink2(vec2(2.0, 0.0))), |ui| ui.horizontal_centered(|ui| imfram::dock::grip(ui)).response.rect.width())
                    .inner;
                let name = "timeline: sequence";
                let col = if focused { v.strong_text_color() } else { v.text_color() };
                ui.painter().text(bar.left_center() + vec2(8.0 + gw, 0.0), Align2::LEFT_CENTER, name, FontId::proportional(12.0), col);
                self.timeline(ui, Rect::from_min_max(pos2(rect.min.x, bar.max.y), rect.max));
                // A scrub there reaches the decoders this redraw.
                self.feed();
            }
            Pane::Wobble => {
                let bar = Rect::from_min_size(rect.min, vec2(rect.width(), TITLE_H));
                let v = ui.visuals().clone();
                ui.painter().rect_filled(bar, 0.0, v.widgets.noninteractive.bg_fill);
                let gw = ui
                    .scope_builder(egui::UiBuilder::new().max_rect(bar.shrink2(vec2(2.0, 0.0))), |ui| ui.horizontal_centered(|ui| imfram::dock::grip(ui)).response.rect.width())
                    .inner;
                ui.painter().text(bar.left_center() + vec2(8.0 + gw, 0.0), Align2::LEFT_CENTER, "wobble: the wob clip under the playhead", FontId::proportional(12.0), v.text_color());
                #[cfg(feature = "fold")]
                {
                    let (mut back, mut own) = (false, false);
                    // (The dock is out on loan while panels draw: the flag.)
                    let main = self.fold.dived;
                    ui.scope_builder(egui::UiBuilder::new().max_rect(bar.shrink2(vec2(6.0, 1.0))).layout(egui::Layout::right_to_left(egui::Align::Center)), |ui| {
                        if main {
                            back = ui.small_button("back to scrub  (cmd shift W)").clicked();
                            own = ui.small_button("own window").clicked();
                        }
                    });
                    if back {
                        self.fold.ask = 1;
                    }
                    if own {
                        self.fold.ask = 2;
                    }
                    self.fold_pane(ui, Rect::from_min_max(pos2(rect.min.x, bar.max.y), rect.max));
                }
            }
            Pane::Controls => {
                let bar = Rect::from_min_size(rect.min, vec2(rect.width(), TITLE_H));
                let v = ui.visuals().clone();
                ui.painter().rect_filled(bar, 0.0, v.widgets.noninteractive.bg_fill);
                let gw = ui
                    .scope_builder(egui::UiBuilder::new().max_rect(bar.shrink2(vec2(2.0, 0.0))), |ui| ui.horizontal_centered(|ui| imfram::dock::grip(ui)).response.rect.width())
                    .inner;
                ui.painter().text(bar.left_center() + vec2(8.0 + gw, 0.0), Align2::LEFT_CENTER, "wob controls", FontId::proportional(12.0), v.text_color());
                #[cfg(feature = "fold")]
                self.fold_controls(ui, Rect::from_min_max(pos2(rect.min.x, bar.max.y), rect.max).shrink(4.0));
            }
            Pane::Source | Pane::Record => {
                if pane == Pane::Source {
                    self.source_viewer(ui, rect);
                } else {
                    self.record_viewer(ui, rect);
                }
                self.hints_over(ui, pane, rect);
            }
        }
    }

    /// The key hints along the top of a viewer (H or the bar's keys button
    /// hides them): the keys split into even rows over the viewers in this
    /// window (two rows over each, four over one alone), so no cap is cut
    /// off. Wobble's motion: hovering a row fans its words out one after
    /// another (and back in), hiding / showing unfolds the rows top down,
    /// and the rows under a row whose words are out dim so the words read.
    fn hints_over(&mut self, ui: &egui::Ui, pane: crate::panes::Pane, rect: Rect) {
        use crate::panes::Pane;
        let bottom_id = egui::Id::new(("hint-bottom", rect.min.x as i32));
        let anim_id = egui::Id::new(("hint-anim", pane as u8));
        let on = self.panes.hints;
        let mut anim: HintAnim = ui.ctx().data(|d| d.get_temp(anim_id)).unwrap_or(HintAnim { vis: if on { 1.0 } else { 0.0 }, rows: Vec::new() });
        let k = 1.0 - (-ui.input(|i| i.stable_dt).min(0.1) * 12.0).exp();
        let mut moving = approach(&mut anim.vis, if on { 1.0 } else { 0.0 }, k);
        if anim.vis <= 0.0 {
            anim.rows.clear();
            ui.ctx().data_mut(|d| {
                d.insert_temp(bottom_id, TITLE_H);
                d.insert_temp(anim_id, anim);
            });
            return;
        }
        let mut views: Vec<Pane> = self.panes.in_host(self.panes.host).into_iter().map(|(p, _)| p).filter(|p| matches!(p, Pane::Source | Pane::Record)).collect();
        if !views.contains(&pane) {
            views.push(pane);
        }
        views.sort_by_key(|p| *p as u8);
        let me = views.iter().position(|p| *p == pane).unwrap_or(0);
        // This viewer's share of the keys, wrapped into as many rows as its
        // width needs.
        let n = views.len().clamp(1, 2);
        let keys = if self.mode == Mode::Selects { crate::selects::KEYS } else { KEYS };
        let per = keys.len().div_ceil(n);
        let mine = &keys[(me * per).min(keys.len())..((me + 1) * per).min(keys.len())];
        let area = Rect::from_min_max(pos2(rect.min.x, rect.min.y + TITLE_H), rect.max);
        let painter = ui.painter_at(area);
        let items = hudkeys::items(ui, mine);
        let mut rows: Vec<hudkeys::TopStrip> = Vec::new();
        let (mut start, mut y) = (0, area.min.y);
        while start < items.len() {
            let mut end = start + 1;
            while end < items.len() {
                let probe = hudkeys::TopStrip::new(&painter, Rect::from_min_max(pos2(area.min.x, y), area.max), &items[start..=end]);
                if probe.row().max.x > area.max.x - 8.0 {
                    break;
                }
                end += 1;
            }
            let strip = hudkeys::TopStrip::new(&painter, Rect::from_min_max(pos2(area.min.x, y), area.max), &items[start..end]);
            y += strip.row().height() + HINT_GAP;
            rows.push(strip);
            start = end;
        }
        anim.rows.resize(rows.len(), 0.0);
        // Each row's share of the show / hide, a beat after the row above
        // (so hiding folds them away bottom up).
        const ROW_STAGGER: f32 = 0.2;
        let nr = rows.len() as f32;
        let show: Vec<f32> = (0..rows.len()).map(|i| ease((anim.vis * (1.0 + ROW_STAGGER * nr) - ROW_STAGGER * i as f32).clamp(0.0, 1.0))).collect();
        // The row the hand is on: its caps, or (once they are out) where
        // its words land, so moving down to read one keeps it up. A row's
        // own caps win, so the rows under an open one stay in reach.
        let pointer = if on && ui.rect_contains_pointer(area) { ui.ctx().pointer_hover_pos() } else { None };
        let hot = pointer.and_then(|p| {
            rows.iter().position(|s| s.row().expand2(vec2(6.0, 3.0)).contains(p)).or_else(|| {
                (0..rows.len()).filter(|&i| anim.rows[i] > 0.3 && rows[i].fan(&painter).contains(p)).max_by(|&a, &b| anim.rows[a].total_cmp(&anim.rows[b]))
            })
        });
        for (i, r) in anim.rows.iter_mut().enumerate() {
            moving |= approach(r, if hot == Some(i) { 1.0 } else { 0.0 }, k);
        }
        // Every row's plate and caps first, the words (they fan downward)
        // last, so no row hides another's.
        for (i, strip) in rows.iter().enumerate() {
            let s = show[i];
            if s <= 0.0 {
                continue;
            }
            let plate = strip.row().expand2(vec2(4.0, 3.0));
            // Fading in, the row unfolds left to right.
            let wipe = Rect::from_min_max(plate.min - vec2(2.0, 2.0), pos2(plate.min.x - 2.0 + (plate.width() + 4.0) * s, plate.max.y + 2.0));
            let mut p = painter.with_clip_rect(wipe);
            p.multiply_opacity(s);
            p.rect_filled(plate, 4.0, Color32::from_black_alpha(170));
            strip.draw_caps(&p);
            // Under a row whose words are out: dimmed, so its words read.
            let dim = anim.rows[..i].iter().fold(0.0f32, |a, r| a.max(*r));
            if dim > 0.0 {
                p.rect_filled(plate.expand(1.0), 4.0, Color32::from_black_alpha((200.0 * dim) as u8));
            }
        }
        let mut out: Vec<usize> = (0..rows.len()).filter(|&i| anim.rows[i] > 0.001).collect();
        out.sort_by_key(|&i| hot == Some(i));
        for i in out {
            let mut p = painter.clone();
            p.multiply_opacity(show[i]);
            rows[i].draw_words(&p, anim.rows[i]);
        }
        // Where the hints end: the readouts sit under them (and follow them
        // up and down as they hide and show).
        let full = rows.last().map_or(area.min.y, |s| s.row().max.y) - rect.min.y;
        let bottom = TITLE_H + (full - TITLE_H) * ease(anim.vis);
        if moving {
            ui.ctx().request_repaint();
        }
        ui.ctx().data_mut(|d| {
            d.insert_temp(bottom_id, bottom);
            d.insert_temp(anim_id, anim);
        });
    }

    /// The dock's state as the panels' API reads it (open, solo).
    pub(crate) fn dock_sync(&mut self, d: &crate::panes::Docks) {
        self.panes.open = d.open_tabs();
        let h = self.panes.host.min(d.hosts.len().saturating_sub(1));
        self.panes.solo = d.soloed(h).or_else(|| d.soloed(crate::panes::MAIN));
    }

    /// W / the view menu: a panel closed or opened (back where it was).
    pub(crate) fn set_pane_open(&mut self, p: crate::panes::Pane, open: bool) {
        if let Some(d) = &mut self.docks {
            for i in 0..d.hosts.len() {
                d.unsolo(i);
            }
            if !open {
                if let Some(i) = d.host_of(p) {
                    self.closed.retain(|(q, _, _)| *q != p);
                    self.closed.push((p, i, d.hosts[i].tree.clone()));
                }
                d.set_open(p, false);
            } else if d.host_of(p).is_none() {
                // Back where it was: that window's layout as it was, when
                // the window still holds just what it held then (bar p).
                let back = self.closed.iter().position(|(q, _, _)| *q == p).map(|k| self.closed.remove(k));
                let put = back.and_then(|(_, i, mut t)| {
                    let now = d.hosts.get(i)?.tree.tabs();
                    for x in t.tabs() {
                        if x != p && !now.contains(&x) {
                            t.remove(x);
                        }
                    }
                    let had = t.tabs();
                    (now.iter().all(|x| had.contains(x)) && t.find(p).is_some()).then_some((i, t))
                });
                match put {
                    Some((i, t)) => d.hosts[i].tree = t,
                    None => d.set_open(p, true),
                }
            }
            let l = d.save();
            self.panes.open = d.open_tabs();
            self.panes.solo = None;
            self.panes.dock = Some(l);
            self.panes.changed();
        }
    }

    /// EXPERIMENT (fold): dive into wobble: it takes the whole main window
    /// (unless it is already up in a window of its own).
    #[cfg(feature = "fold")]
    pub(crate) fn fold_open_pane(&mut self) {
        use crate::panes::{Pane, MAIN};
        self.fold.dived = true;
        self.closed.retain(|(q, _, _)| *q != Pane::Wobble);
        if let Some(d) = &mut self.docks {
            if d.host_of(Pane::Wobble).is_none() {
                d.unsolo(MAIN);
                d.hosts[MAIN].tree.split(imfram::dock::Side::Left, 0.42, vec![Pane::Wobble]);
            }
        }
        self.fold_layout();
    }

    /// The wob controls panel open: the first time, a column down the left
    /// of the main window (drag its grip to put it elsewhere).
    #[cfg(feature = "fold")]
    pub(crate) fn fold_open_controls(&mut self) {
        use crate::panes::{Pane, MAIN};
        let Some(d) = &mut self.docks else { return };
        if d.host_of(Pane::Controls).is_some() {
            return;
        }
        if self.closed.iter().any(|(q, _, _)| *q == Pane::Controls) {
            self.set_pane_open(Pane::Controls, true);
            return;
        }
        d.unsolo(MAIN);
        d.hosts[MAIN].tree.split(imfram::dock::Side::Left, 0.8, vec![Pane::Controls]);
        self.panes.open = d.open_tabs();
    }

    /// Wobble is never side by side with Scrub: in the main window it is
    /// alone (the dock's solo), and only while dived; otherwise it is in a
    /// window of its own, or closed. Kept true every frame (a saved layout,
    /// a torn-off window closed and put back).
    #[cfg(feature = "fold")]
    pub(crate) fn fold_layout(&mut self) {
        use crate::panes::{Pane, MAIN};
        let Some(d) = &mut self.docks else { return };
        if d.host_of(Pane::Wobble) != Some(MAIN) {
            self.fold.popping = 0;
            return;
        }
        if self.fold.popping > 0 {
            self.fold.popping -= 1;
            return;
        }
        if !self.fold.dived {
            self.fold_close_pane();
        } else if d.soloed(MAIN) != Some(Pane::Wobble) {
            d.unsolo(MAIN);
            d.solo(Pane::Wobble);
            self.panes.solo = Some(Pane::Wobble);
        }
    }

    /// Back to Scrub: the wobble panel closed, the layout as it was.
    #[cfg(feature = "fold")]
    pub(crate) fn fold_close_pane(&mut self) {
        use crate::panes::{Pane, MAIN};
        self.fold.dived = false;
        if let Some(d) = &mut self.docks {
            if d.host_of(Pane::Wobble) == Some(MAIN) {
                d.unsolo(MAIN);
                d.set_open(Pane::Wobble, false);
                self.panes.open = d.open_tabs();
                self.panes.solo = None;
            }
        }
    }

    /// Wobble in a window of its own (for a second monitor).
    #[cfg(feature = "fold")]
    pub(crate) fn fold_own_window(&mut self) {
        use crate::panes::{Pane, MAIN};
        if let Some(d) = &mut self.docks {
            if d.host_of(Pane::Wobble).is_none() {
                d.hosts[MAIN].tree.split(imfram::dock::Side::Left, 0.42, vec![Pane::Wobble]);
            }
            d.unsolo(MAIN);
            self.panes.solo = None;
            d.asks.push(imfram::dock::Ask::Detach(Pane::Wobble));
        }
        self.fold.dived = false;
        self.fold.popping = 120;
    }

    /// Where wobble is: 0 closed, 1 the whole main window, 2 its own window.
    #[cfg(feature = "fold")]
    pub(crate) fn fold_where(&self) -> u8 {
        match self.docks.as_ref().and_then(|d| d.host_of(crate::panes::Pane::Wobble)) {
            None => 0,
            Some(crate::panes::MAIN) => 1,
            Some(_) => 2,
        }
    }

    /// shift+W: every panel open. Not wobble or its controls (the `fold`
    /// feature): wobble is never side by side with Scrub, and both open on
    /// their own gestures (cmd shift W, a wob clip).
    pub(crate) fn show_all_panes(&mut self) {
        use crate::panes::Pane;
        for p in crate::panes::ALL {
            if matches!(p, Pane::Wobble | Pane::Controls) {
                continue;
            }
            if !self.panes.open.contains(&p) {
                self.set_pane_open(p, true);
            }
        }
        if let Some(d) = &mut self.docks {
            for i in 0..d.hosts.len() {
                d.unsolo(i);
            }
        }
        self.panes.solo = None;
    }

    /// shift+R: the two viewers the same size (where they share a split).
    pub(crate) fn match_viewers(&mut self) {
        if let Some(d) = &mut self.docks {
            for h in &mut d.hosts {
                crate::panes::match_viewers(&mut h.tree);
            }
        }
    }

    /// The main window joins the dock as its external host, with the
    /// default layout (the saved one is restored once windows can be made).
    pub fn attach_window(&mut self, window: Arc<winit::window::Window>, ctx: egui::Context, headless: bool) {
        let mut d = crate::panes::Docks::new(media::sys::APP, headless);
        d.add_external(crate::panes::Role::Main, window, ctx, self.panes.default_tree());
        self.dock_restore = self.panes.take_saved();
        self.panes.open = d.open_tabs();
        self.docks = Some(d);
    }

    /// Between frames, where windows can be made: a saved layout restored,
    /// a reset, the dock's asks (detach, move, close) taken up, and every
    /// window holding a viewer given its GPU state.
    pub fn dock_between(&mut self, el: &winit::event_loop::ActiveEventLoop, gpu: &imfram::shell::Gpu) {
        let Some(mut d) = self.docks.take() else { return };
        use crate::panes::{Role, MAIN};
        if let Some(saved) = self.dock_restore.take() {
            let (main, rest): (Vec<_>, Vec<_>) = saved.windows.into_iter().partition(|w| w.role == Role::Main);
            if let Some(m) = main.into_iter().next().filter(|m| !m.tree.is_empty()) {
                d.hosts[MAIN].tree = m.tree;
            }
            d.restore(el, gpu, imfram::dock::Layout { windows: rest });
        }
        if std::mem::take(&mut self.dock_reset) {
            for i in (0..d.hosts.len()).rev() {
                if d.hosts[i].role == Role::Detached {
                    d.close_requested(i);
                }
            }
            d.unsolo(MAIN);
            d.hosts[MAIN].tree = self.panes.default_tree();
        }
        let n0 = d.hosts.len();
        d.apply(Some(el), gpu);
        // Every panel somewhere: one a saved file lost opens at home.
        for p in crate::panes::ALL {
            if d.host_of(p).is_none() && !self.panes.open.is_empty() && self.panes.open.contains(&p) {
                d.set_open(p, true);
            }
        }
        if d.hosts.len() < n0 {
            self.panes.forget_hosts_from(d.hosts.len());
        }
        let fmt = |w: &imfram::shell::UiWindow| w.format();
        for h in &mut d.hosts {
            if let Some(w) = h.win.own() {
                let f = fmt(w);
                let res = &mut w.layer_mut().renderer_mut().callback_resources;
                if res.get::<crate::viewer::Viewers>().is_none() {
                    res.insert(crate::viewer::Viewers::new(&gpu.device, f, 3));
                    #[cfg(feature = "fold")]
                    res.insert(crate::viewer::TexBlit::new(&gpu.device, f));
                    style(w.ctx());
                }
            }
        }
        d.poll();
        self.dock_sync(&d);
        self.docks = Some(d);
    }

    /// A torn-off window's event: drawn through the dock when it asks.
    /// False: not one of ours.
    pub fn dock_event(&mut self, gpu: &imfram::shell::Gpu, id: winit::window::WindowId, event: &winit::event::WindowEvent) -> bool {
        let Some(mut d) = self.docks.take() else { return false };
        let Some(i) = d.by_window(id).filter(|&i| !d.is_external(i)) else {
            self.docks = Some(d);
            return false;
        };
        if d.event(gpu, i, event) {
            self.panes.begin_host(i);
            d.frame(i, gpu, self, None);
            self.panes.end_host(i);
            self.panes.host = crate::panes::MAIN;
            let released = true;
            self.dock_sync(&d);
            self.panes.note_layout(&d, released);
        }
        self.docks = Some(d);
        true
    }

    /// Ask every torn-off window to draw again (the main one just did).
    pub fn dock_repaint(&mut self) {
        if let Some(d) = &mut self.docks {
            for h in &mut d.hosts {
                if !h.win.is_external() {
                    h.win.request_repaint();
                }
            }
        }
    }

    /// Draw window `i` offscreen through the dock (tests).
    pub fn dock_offscreen(&mut self, i: usize, gpu: &imfram::shell::Gpu) {
        self.dock_offscreen_with(i, gpu, vec![]);
    }

    /// `dock_offscreen` with input events for that window (tests).
    pub fn dock_offscreen_with(&mut self, i: usize, gpu: &imfram::shell::Gpu, events: Vec<egui::Event>) {
        let Some(mut d) = self.docks.take() else { return };
        if i < d.hosts.len() && !d.is_external(i) {
            self.panes.begin_host(i);
            let _ = d.frame(i, gpu, self, Some(events));
            self.panes.end_host(i);
            self.panes.host = crate::panes::MAIN;
            self.dock_sync(&d);
        }
        self.docks = Some(d);
    }

    /// window > reset layout (tests).
    pub fn dock_reset_now(&mut self) {
        self.dock_reset = true;
    }

    /// Torn-off windows (tests).
    pub fn dock_windows(&self) -> usize {
        self.docks.as_ref().map_or(0, |d| d.hosts.len())
    }

    /// Ask the dock (tests): detach a panel into a window of its own.
    pub fn dock_detach(&mut self, p: crate::panes::Pane) {
        if let Some(d) = &mut self.docks {
            d.asks.push(imfram::dock::Ask::Detach(p));
        }
    }

    /// The window a panel is in (tests).
    pub fn dock_host_of(&self, p: crate::panes::Pane) -> Option<usize> {
        self.docks.as_ref().and_then(|d| d.host_of(p))
    }
}

impl imfram::dock::Panels<crate::panes::Pane> for Editor {
    fn begin(&mut self, ui: &mut egui::Ui) {
        // A torn-off window: the app's keys work there too (the main
        // window ran them before its dock).
        if self.panes.host != crate::panes::MAIN {
            imfram::theme::sync(ui.ctx());
            self.keys(ui);
        }
    }

    // (The main window has the menubar; a torn-off window's panels stand
    // alone. Closed panels come back from the view menu.)
    fn bar_shown(&self) -> bool {
        false
    }

    fn tab(&mut self, ui: &mut egui::Ui, pane: crate::panes::Pane) {
        self.tab_body(ui, pane);
    }

    fn end(&mut self, ui: &mut egui::Ui) {
        if self.panes.host != crate::panes::MAIN {
            let busy = self.rec.transport.playing() || self.source.as_ref().is_some_and(|c| c.transport.playing());
            if busy {
                ui.ctx().request_repaint();
            }
        }
    }

    fn scroll_bars(&self, _tab: &crate::panes::Pane) -> [bool; 2] {
        [false, false]
    }

    // No row of tab headers: each panel's top row starts with a grip
    // (`dock::grip`) that drags, stacks and has the panel's menu.
    fn tab_bars(&self) -> bool {
        false
    }

    fn solo_key(&self) -> Option<egui::Key> {
        Some(egui::Key::Q)
    }
}
