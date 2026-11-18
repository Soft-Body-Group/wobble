//! Scrub: a fast, hotkey-driven video editor. Step 1: one viewer that plays
//! and scrubs a movie or an image sequence.
//!
//! ```text
//! scrub [file]          open a movie, or a sequence from one of its frames
//! scrub file --shot out.png [--frame N]   the window, offscreen, to a PNG
//! scrub file --play SECS [--speed S]       headless playback at 60 Hz, measured
//! scrub --post POSTDIR [--rushes DIR]...    open an organisicator advanced post (how orgo launches Scrub)
//! scrub file --cuts N [--play S | --shot P] cut the file into N clips on the timeline (--fx: on one track, with speed changes and dissolves), play / shoot the record side
//! scrub file --optimise [--dest beside|cache] [--profile lt|standard|hq]
//!                                          transcode to ProRes, no window
//! ```
//!
//! | module      | what it owns                                              |
//! |-------------|-----------------------------------------------------------|
//! | `viewer`    | the GPU side: planes up, YCbCr -> RGB, draw fit to view    |
//! | `transport` | the playhead: J/K/L shuttle, steps, the clock, drops      |
//! | `editor`    | the window's UI: viewer, HUD, scrub bar, keys, browser    |
//! | `optimiseui`| the optimise dialog, job, HUD line and cache window       |

mod apiserve;
mod appicon;
mod audio;
mod bin;
mod binthumbs;
#[cfg(feature = "clips")]
mod clipbrowser;
mod editor;
#[cfg(feature = "fold")]
mod fold;
mod exportui;
mod feedback;
mod edits;
mod marketing;
mod mouse;
mod panes;
mod optimiseui;
mod organisicator;
mod program;
mod recorder;
mod project;
mod render;
mod testapi;
mod text;
mod textui;
mod testtools;
mod timelineui;
mod tour;
mod transport;
mod viewer;
mod viewnav;
#[cfg(feature = "wobble")]
mod testwobble;
#[cfg(feature = "wobble")]
mod wobbleclip;

use imfram::shell::{Gpu, UiWindow};
use std::path::PathBuf;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::WindowId;

struct App {
    gpu: Option<Gpu>,
    win: Option<UiWindow>,
    editor: editor::Editor,
    /// `--shot PNG`: render the window offscreen once the picture is in, save, exit.
    shot: Option<PathBuf>,
    /// `--test-optimise`: start an optimise in the app, wait, check the switch.
    test_optimise: bool,
    /// `--test-keys`: drive the editing hotkeys through real key events.
    test_keys: bool,
    /// `--test-mouse`: drive timeline drags through real pointer events.
    test_mouse: bool,
    /// `--test-tools`: the editing tools, audio tracks, drops (testtools.rs).
    test_tools: bool,
    /// `--test-api`: the scripting API and the MCP bridge end to end (testapi.rs).
    test_api: bool,
    /// `--test-wobble` (the `wobble` feature): Wobble clips end to end.
    test_wobble: bool,
    /// `--test-project`: save, reopen, compare.
    test_project: bool,
    test_colour: bool,
    test_record: bool,
    test_realtime: bool,
    title: String,
    /// `--play SECS [--speed S]`: play offscreen at a paced 60 Hz, report.
    play: Option<(f64, f64)>,
}

impl App {
    /// Timeline mouse editing through real pointer events: move a clip (its
    /// sound follows), move it up two tracks, trim an edge, snap to the
    /// playhead, cancel with Esc, and every drag one undo step.
    fn test_mouse(&mut self, el: &ActiveEventLoop) {
        use egui::{Event as E, Modifiers as M, PointerButton as B};
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let ed = &mut self.editor;
        ed.build_cuts(4, false); // V1 at 0 and 24, V2 at 12 and 36, 12 frames each
        ed.focus_record();
        for _ in 0..3 {
            let _ = win.offscreen(gpu, ed);
        }
        let mut fails: Vec<String> = Vec::new();
        let frame = |win: &mut UiWindow, ed: &mut editor::Editor, evs: Vec<E>| {
            win.inject(evs);
            let _ = win.offscreen(gpu, ed);
        };
        // Press at `from`, move in steps to `to`, optionally press keys, release.
        let drag = |win: &mut UiWindow, ed: &mut editor::Editor, from: egui::Pos2, to: egui::Pos2, mods: M, keys: Vec<E>| {
            frame(win, ed, vec![E::PointerMoved(from)]);
            frame(win, ed, vec![E::PointerButton { pos: from, button: B::Primary, pressed: true, modifiers: mods }]);
            for k in 1..=8 {
                let p = from + (to - from) * (k as f32 / 8.0);
                frame(win, ed, vec![E::PointerMoved(p)]);
            }
            if !keys.is_empty() {
                frame(win, ed, keys);
            }
            frame(win, ed, vec![E::PointerButton { pos: to, button: B::Primary, pressed: false, modifiers: mods }]);
            frame(win, ed, vec![]);
        };
        let has = |ed: &editor::Editor, c: (usize, i64, i64)| ed.sequence_summary().clips.iter().any(|x| (x.0, x.1, x.2) == c);
        let pt = |ed: &editor::Editor, t: usize, f: f64| ed.timeline_point(t, f).expect("lane");
        ed.set_snap(false);
        if ed.ripple_trims() {
            fails.push("ripple trims are on by default".into());
        }
        let audio = ed.sequence_has_audio();

        // 1. The V1 clip at 0, six frames later (body press, middle of the clip).
        let (a, b) = (pt(ed, model::V1, 6.0), pt(ed, model::V1, 12.0));
        drag(win, ed, a, b, M::NONE, vec![]);
        if !has(ed, (model::V1, 6, 12)) {
            fails.push(format!("move +6: {:?}", ed.sequence_summary().clips));
        }
        if audio && !has(ed, (model::A1, 6, 12)) {
            fails.push("the linked sound did not move with it".into());
        }
        // One undo puts it back.
        ed.undo_once();
        if !has(ed, (model::V1, 0, 12)) {
            fails.push("one undo did not undo the whole drag".into());
        }
        let _ = win.offscreen(gpu, ed);

        // 2. Up two rows: V1 -> V3.
        let (a, b) = (pt(ed, model::V1, 6.0), pt(ed, model::V3, 6.0));
        drag(win, ed, a, b, M::NONE, vec![]);
        if !has(ed, (model::V3, 0, 12)) {
            fails.push(format!("move to V3: {:?}", ed.sequence_summary().clips));
        }
        ed.undo_once();
        let _ = win.offscreen(gpu, ed);

        // 3. The out edge of the V1 clip at 24, three frames earlier.
        let (a, b) = (pt(ed, model::V1, 35.9), pt(ed, model::V1, 32.9));
        drag(win, ed, a, b, M::NONE, vec![]);
        if !ed.sequence_summary().clips.iter().any(|c| c.0 == model::V1 && c.1 == 24 && c.2 == 9) {
            fails.push(format!("trim out -3: {:?}", ed.sequence_summary().clips));
        }
        ed.undo_once();
        let _ = win.offscreen(gpu, ed);

        // 3b. The IN edge of that clip, plainly (ripple is off by default):
        // the cut moves three frames later with the source in, a gap opens
        // before it, and nothing else moves.
        let before = ed.sequence_summary().clips;
        let src0 = before.iter().find(|c| c.0 == model::V1 && c.1 == 24).map(|c| c.3);
        let (a, b) = (pt(ed, model::V1, 24.1), pt(ed, model::V1, 27.1));
        drag(win, ed, a, b, M::NONE, vec![]);
        let after = ed.sequence_summary().clips;
        let moved = after.iter().find(|c| c.0 == model::V1 && c.1 == 27).map(|c| (c.2, c.3));
        let others_same = before.iter().filter(|c| !(c.1 == 24 && (c.0 == model::V1 || c.0 == model::A1))).all(|c| after.contains(c));
        if moved != src0.map(|s| (9, s + 3)) || !others_same {
            fails.push(format!("plain left-edge trim: {before:?} -> {after:?}"));
        }
        ed.undo_once();
        let _ = win.offscreen(gpu, ed);

        // 4. Snapping: playhead at 30, the clip at 24 dragged to start near it.
        ed.set_snap(true);
        ed.seek_record(30);
        let _ = win.offscreen(gpu, ed);
        let fpp = ed.timeline_fpp();
        let near = 30.0 + 3.0 * fpp; // three pixels past the playhead
        let (a, b) = (pt(ed, model::V1, 30.0), pt(ed, model::V1, 30.0 + (near - 24.0)));
        drag(win, ed, a, b, M::NONE, vec![]);
        if !has(ed, (model::V1, 30, 12)) {
            fails.push(format!("snap to the playhead: {:?}", ed.sequence_summary().clips));
        }
        ed.undo_once();
        let _ = win.offscreen(gpu, ed);

        // 5. Esc mid-drag puts everything back and makes no undo step.
        let before = ed.sequence_summary().clips;
        let (a, b) = (pt(ed, model::V1, 6.0), pt(ed, model::V1, 20.0));
        let esc = vec![E::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: M::NONE }];
        drag(win, ed, a, b, M::NONE, esc);
        if ed.sequence_summary().clips != before || ed.dragging() {
            fails.push("Esc did not cancel the drag".into());
        }
        // 6. cmd held: an insert move. The V1 clip at 24 dropped at 6 cuts
        // the clip at 0 and pushes its tail on.
        ed.set_snap(false);
        let (a, b) = (pt(ed, model::V1, 30.0), pt(ed, model::V1, 12.0));
        drag(win, ed, a, b, M::COMMAND, vec![]);
        if !(has(ed, (model::V1, 6, 12)) && has(ed, (model::V1, 0, 6)) && has(ed, (model::V1, 18, 6))) {
            fails.push(format!("insert move: {:?}", ed.sequence_summary().clips));
        }
        ed.undo_once();
        let _ = win.offscreen(gpu, ed);

        // 7. Make a cut on V1 (the clip at 24 moved to 12), put a dissolve
        // on it, then drag the dissolve's end edge two frames later.
        let (a, b) = (pt(ed, model::V1, 30.0), pt(ed, model::V1, 18.0));
        drag(win, ed, a, b, M::NONE, vec![]);
        ed.seek_record(12);
        ed.select_edit_at(model::V1, 12);
        ed.add_dissolve_len(4);
        let _ = win.offscreen(gpu, ed);
        let d0 = ed.sequence_summary().dissolves.first().copied();
        if let Some((_, at, _, after)) = d0 {
            let (a, b) = (pt(ed, model::V1, (at + after) as f64), pt(ed, model::V1, (at + after + 2) as f64));
            drag(win, ed, a, b, M::NONE, vec![]);
            let d1 = ed.sequence_summary().dissolves.first().copied();
            if d1.map(|d| d.3) != Some(after + 2) {
                fails.push(format!("dissolve edge: {d0:?} -> {d1:?}"));
            }
        } else {
            fails.push("no dissolve to drag".into());
        }

        // 8. The roll tool (N) rolls the cut; V goes back to select.
        frame(win, ed, vec![E::Key { key: egui::Key::N, physical_key: None, pressed: true, repeat: false, modifiers: M::NONE }]);
        let (a, b) = (pt(ed, model::V1, 11.9), pt(ed, model::V1, 14.9));
        drag(win, ed, a, b, M::NONE, vec![]);
        frame(win, ed, vec![E::Key { key: egui::Key::V, physical_key: None, pressed: true, repeat: false, modifiers: M::NONE }]);
        if !(has(ed, (model::V1, 0, 15)) && ed.sequence_summary().clips.iter().any(|c| c.0 == model::V1 && c.1 == 15 && c.2 == 9)) {
            fails.push(format!("roll by drag: {:?}", ed.sequence_summary().clips));
        }
        // 9. Panels on the dock: W over the bin closes it, shift+W brings
        // it back where it was, Q over the timeline shows its leaf alone
        // and Q puts the layout back, and the divider right of the bin
        // drags.
        let key = |win: &mut UiWindow, ed: &mut editor::Editor, at: egui::Pos2, k: egui::Key, m: M| {
            frame(win, ed, vec![E::PointerMoved(at)]);
            frame(win, ed, vec![E::Key { key: k, physical_key: None, pressed: true, repeat: false, modifiers: m }]);
            frame(win, ed, vec![E::Key { key: k, physical_key: None, pressed: false, repeat: false, modifiers: m }]);
            // Let the leaves glide home (wall clock: the dock eases by time).
            let t = std::time::Instant::now();
            while t.elapsed().as_millis() < 700 {
                frame(win, ed, vec![]);
                std::thread::sleep(std::time::Duration::from_millis(8));
            }
        };
        let bin0 = ed.pane_rect(panes::Pane::Bin).unwrap_or(egui::Rect::NOTHING);
        key(win, ed, bin0.center(), egui::Key::W, M::NONE);
        if ed.pane_shown(panes::Pane::Bin) {
            fails.push("W did not close the bin".into());
        }
        key(win, ed, egui::pos2(800.0, 500.0), egui::Key::W, M::SHIFT);
        let bin1 = ed.pane_rect(panes::Pane::Bin).unwrap_or(egui::Rect::NOTHING);
        if !ed.pane_shown(panes::Pane::Bin) || (bin1.min - bin0.min).length() > 2.0 || (bin1.width() - bin0.width()).abs() > 2.0 {
            fails.push(format!("shift W did not bring the bin back where it was: {bin0:?} -> {bin1:?}"));
        }
        let tl_pt = pt(ed, model::V1, 2.0);
        key(win, ed, tl_pt, egui::Key::Q, M::NONE);
        if ed.pane_shown(panes::Pane::Bin) || !ed.pane_shown(panes::Pane::Timeline) {
            fails.push("Q did not show the timeline alone".into());
        }
        key(win, ed, tl_pt, egui::Key::Q, M::NONE);
        if !ed.pane_shown(panes::Pane::Bin) || !ed.pane_shown(panes::Pane::Source) {
            fails.push("Q again did not put the panels back".into());
        }
        let w0 = ed.bin_width();
        let b = ed.pane_rect(panes::Pane::Bin).unwrap_or(egui::Rect::NOTHING);
        let sx = b.max.x + 2.0;
        drag(win, ed, egui::pos2(sx, b.center().y), egui::pos2(sx + 80.0, b.center().y), M::NONE, vec![]);
        let t = std::time::Instant::now();
        while t.elapsed().as_millis() < 700 {
            frame(win, ed, vec![]);
            std::thread::sleep(std::time::Duration::from_millis(8));
        }
        if (ed.bin_width() - (w0 + 80.0)).abs() > 4.0 {
            fails.push(format!("the divider right of the bin: {w0} -> {}", ed.bin_width()));
        }
        // 10. Tear the record viewer off into a window of its own: it draws
        // there, the main window keeps the rest, and playing it there
        // drops nothing and shows as many pictures as docked.
        let play = |win: &mut UiWindow, ed: &mut editor::Editor, torn: bool| -> (f64, u64) {
            ed.focus_record();
            ed.seek_record(0);
            frame(win, ed, vec![]);
            if torn {
                ed.dock_offscreen(1, gpu);
            }
            ed.wait_for_picture(std::time::Duration::from_secs(10));
            let (base, drop0) = (ed.shown_total(), ed.dropped());
            ed.play(1.0);
            let t0 = std::time::Instant::now();
            let mut n = 0u64;
            while t0.elapsed().as_secs_f64() < 1.6 {
                let due = t0 + std::time::Duration::from_secs_f64((n + 1) as f64 / 60.0);
                frame(win, ed, vec![]);
                if torn {
                    ed.dock_offscreen(1, gpu);
                }
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                n += 1;
                std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
            }
            ed.play(0.0);
            ((ed.shown_total() - base) as f64 / t0.elapsed().as_secs_f64(), ed.dropped() - drop0)
        };
        let (docked, ddrop) = play(win, ed, false);
        ed.dock_detach(panes::Pane::Record);
        ed.dock_between(el, gpu);
        let host = ed.dock_host_of(panes::Pane::Record);
        if ed.dock_windows() != 2 || host != Some(1) {
            fails.push(format!("tear-off: {} windows, record in {host:?}", ed.dock_windows()));
        }
        let (torn, tdrop) = play(win, ed, true);
        // In the torn-off window too, I over the record viewer marks the
        // record whatever has the focus (its own rects, its own pointer).
        ed.focus_bin();
        ed.set_record_marks(model::Marks::default());
        ed.dock_offscreen(1, gpu);
        if let Some(r) = ed.pane_rect_in(1, panes::Pane::Record) {
            let c = r.center();
            ed.dock_offscreen_with(1, gpu, vec![E::PointerMoved(c)]);
            ed.dock_offscreen_with(1, gpu, vec![E::Key { key: egui::Key::I, physical_key: None, pressed: true, repeat: false, modifiers: M::NONE }]);
            ed.dock_offscreen_with(1, gpu, vec![E::Key { key: egui::Key::I, physical_key: None, pressed: false, repeat: false, modifiers: M::NONE }]);
            if ed.record_marks().inp.is_none() {
                fails.push("I over the torn-off record viewer did not mark the record".into());
            }
        } else {
            fails.push("the torn-off window drew no record viewer".into());
        }
        println!("INFO record viewer playing: docked {docked:.1} pictures/s ({ddrop} dropped), torn off {torn:.1} pictures/s ({tdrop} dropped)");
        if tdrop != 0 || torn < docked * 0.85 {
            fails.push(format!("playing in the torn-off window: {torn:.1} pictures/s against {docked:.1} docked, {tdrop} dropped"));
        }
        // window > reset layout: every panel back in the main window.
        ed.dock_reset_now();
        ed.dock_between(el, gpu);
        for _ in 0..3 {
            frame(win, ed, vec![]);
        }
        // 11. Drag the TIMELINE by its grip (top left of its bar) onto the
        // record viewer's right edge: it lands beside the record viewer.
        let settle = |win: &mut UiWindow, ed: &mut editor::Editor| {
            let t = std::time::Instant::now();
            while t.elapsed().as_millis() < 700 {
                frame(win, ed, vec![]);
                std::thread::sleep(std::time::Duration::from_millis(8));
            }
        };
        let grip = imfram::dock::tab_rect(win.ctx(), panes::Pane::Timeline);
        let rec = ed.pane_rect(panes::Pane::Record).unwrap_or(egui::Rect::NOTHING);
        match grip {
            Some(g) => {
                let tl0 = ed.pane_rect(panes::Pane::Timeline).unwrap_or(egui::Rect::NOTHING);
                if g.min.x > tl0.min.x + 60.0 {
                    fails.push(format!("the timeline's grip is not at its top left: {g:?} in {tl0:?}"));
                }
                let to = egui::pos2(rec.max.x - 20.0, rec.center().y);
                frame(win, ed, vec![E::PointerMoved(g.center())]);
                frame(win, ed, vec![E::PointerButton { pos: g.center(), button: B::Primary, pressed: true, modifiers: M::NONE }]);
                for k in 1..=12 {
                    frame(win, ed, vec![E::PointerMoved(g.center() + (to - g.center()) * (k as f32 / 12.0))]);
                }
                settle(win, ed);
                frame(win, ed, vec![E::PointerButton { pos: to, button: B::Primary, pressed: false, modifiers: M::NONE }]);
                ed.dock_between(el, gpu);
                settle(win, ed);
                let tl = ed.pane_rect(panes::Pane::Timeline).unwrap_or(egui::Rect::NOTHING);
                let rec2 = ed.pane_rect(panes::Pane::Record).unwrap_or(egui::Rect::NOTHING);
                if !(tl.min.x >= rec2.max.x - 1.0 && tl.width() < tl0.width() * 0.6) {
                    fails.push(format!("dragging the timeline's grip onto the record viewer's edge: timeline {tl0:?} -> {tl:?}, record {rec2:?}"));
                }
            }
            None => fails.push("the timeline has no grip".into()),
        }

        ed.dock_reset_now();
        ed.dock_between(el, gpu);
        if ed.dock_windows() != 1 || ed.dock_host_of(panes::Pane::Record) != Some(0) {
            fails.push(format!("reset layout: {} windows, record in {:?}", ed.dock_windows(), ed.dock_host_of(panes::Pane::Record)));
        }
        if fails.is_empty() {
            println!("TEST PASS timeline mouse: move (sound follows), across tracks, trim, plain left-edge trim, snap, Esc, insert move, dissolve edge, roll, one undo per drag; dock: W, shift W, Q, divider, tear-off + play (0 dropped), reset layout, drag the timeline by its grip");
        } else {
            println!("TEST FAIL {}", fails.join(" | "));
        }
    }

    /// Playback keeps wall-clock time: over 5 s (after a 1 s start) the
    /// playhead advances by the clip's frame rate x the time, within 1%,
    /// for ProRes with and without sound, H.264 at 25 and 60 fps with
    /// sound, HEVC, a sequence of 60 fps cuts with sound, and the same with
    /// the sound feeder starved (the clock must not wait on it).
    fn test_realtime(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata");
        let mut fails = Vec::new();
        let mut report = Vec::new();
        let mut measure = |win: &mut UiWindow, ed: &mut editor::Editor, name: &str| {
            ed.wait_for_picture(std::time::Duration::from_secs(10));
            ed.play(1.0);
            let t0 = std::time::Instant::now();
            let mut n = 0u64;
            let mut start: Option<(f64, std::time::Instant)> = None;
            let mut last = (0.0, t0);
            // 5 s of steady playback, from the first redraw past 1 s that is
            // no longer pre-rolling; each reading timed when it was drawn.
            while start.is_none_or(|(_, ts)| ts.elapsed().as_secs_f64() < 5.0) && t0.elapsed().as_secs_f64() < 12.0 {
                let due = t0 + std::time::Duration::from_secs_f64((n + 1) as f64 / 60.0);
                let _ = win.offscreen(gpu, ed);
                let now = (ed.playhead_pos(), std::time::Instant::now());
                if start.is_none() && t0.elapsed().as_secs_f64() >= 1.0 && !ed.prerolling() {
                    start = Some(now);
                }
                last = now;
                let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                n += 1;
                std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
            }
            let (p0, ts) = start.unwrap_or((0.0, t0));
            let rate = (last.0 - p0) / (last.1.duration_since(ts).as_secs_f64() * ed.active_fps()).max(1e-9);
            ed.play(0.0);
            report.push(format!("{name} {rate:.3}x"));
            if (rate - 1.0).abs() > 0.01 {
                fails.push(format!("{name}: playhead at {rate:.3}x real time"));
            }
        };
        let ed = &mut self.editor;
        for f in ["prores_hq_4k_audio.mov", "prores_hq_4k.mov", "h264_1080_audio_tc.mp4", "h264_2700_60.mp4", "hevc_4k_10bit.mov"] {
            let p = dir.join(f);
            if !p.exists() {
                continue;
            }
            ed.open(&p);
            ed.focus_source();
            ed.seek_display(0);
            measure(win, ed, f);
        }
        let cuts = dir.join("h264_2700_60.mp4");
        if cuts.exists() {
            ed.open(&cuts);
            ed.build_cuts(30, false);
            ed.focus_record();
            ed.seek_record(0);
            measure(win, ed, "60 fps cuts with sound");
            std::env::set_var("SCRUB_AUDIO_STARVE_MS", "40");
            ed.seek_record(0);
            measure(win, ed, "the same, sound feeder starved");
            std::env::remove_var("SCRUB_AUDIO_STARVE_MS");
        }
        println!("INFO realtime: {}", report.join(", "));
        if fails.is_empty() {
            println!("TEST PASS realtime: playback keeps wall-clock time within 1% over 5 s ({} cases, incl. sound, 60 fps cuts and a starved sound feeder)", report.len());
        } else {
            println!("TEST FAIL realtime: {}", fails.join(" | "));
        }
    }

    /// The screen recorder (wobble's): a take of the window while the
    /// sequence plays (frames copied as the app copies the presented frame,
    /// here from the offscreen frame), stopped into review with at least 45
    /// frames for ~2.5 s, then saved to an MP4 (the OS's H.264) and a GIF in
    /// `EDIFY_RECORD_DIR`, both read back.
    fn test_record(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let dir = std::env::temp_dir().join(format!("scrub-record-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::env::set_var("SCRUB_RECORD_DIR", &dir);
        let ed = &mut self.editor;
        ed.build_cuts(4, false);
        ed.focus_record();
        ed.wait_for_picture(std::time::Duration::from_secs(10));
        let tex = win.offscreen(gpu, ed);
        let (w, h, fmt) = (tex.width(), tex.height(), tex.format());
        let mut fails = Vec::new();
        {
            let mut r = ed.recorder.lock().unwrap();
            r.opts.gif = true;
            r.start(&gpu.device, [0.0, 0.0, w as f32, h as f32], w, h, win.ctx().pixels_per_point());
            if !r.recording() {
                fails.push(format!("the take did not start: {:?}", r.error));
            }
        }
        ed.play(1.0);
        let t0 = std::time::Instant::now();
        let mut n = 0u64;
        while t0.elapsed().as_secs_f64() < 2.5 {
            let due = t0 + std::time::Duration::from_secs_f64((n + 1) as f64 / 60.0);
            let tex = win.offscreen(gpu, ed);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            let a = n as f32 * 0.1;
            ed.recorder.lock().unwrap().capture(&mut enc, &tex, fmt, Some([w as f32 * (0.5 + 0.3 * a.cos()), h as f32 * (0.5 + 0.3 * a.sin())]));
            gpu.queue.submit([enc.finish()]);
            ed.recorder.lock().unwrap().after_submit();
            n += 1;
            std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
        }
        ed.play(0.0);
        ed.recorder.lock().unwrap().stop(&gpu.device);
        let t1 = std::time::Instant::now();
        let mut frames = None;
        while t1.elapsed().as_secs() < 10 {
            let _ = win.offscreen(gpu, ed);
            frames = ed.recorder.lock().unwrap().review_frames();
            if frames.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        match frames {
            Some(f) if f >= 45 => {}
            other => fails.push(format!("frames in review: {other:?}")),
        }
        ed.recorder.lock().unwrap().save_all();
        let t2 = std::time::Instant::now();
        let mut saved = None;
        while t2.elapsed().as_secs() < 60 {
            let _ = win.offscreen(gpu, ed);
            let r = ed.recorder.lock().unwrap();
            if !r.saving() {
                saved = r.save_result();
                if saved.is_some() {
                    break;
                }
            }
            drop(r);
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let probe = |p: &std::path::Path| -> Option<u64> {
            let o = std::process::Command::new("ffprobe").args(["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"]).arg(p).output().ok()?;
            String::from_utf8_lossy(&o.stdout).trim().parse().ok()
        };
        match saved {
            Some(Ok(s)) => {
                for (p, min) in s.mp4.iter().map(|p| (p, 45)).chain(s.gif.iter().map(|p| (p, 20))) {
                    // (The GIF runs at half the rate, wobble's GIF_FPS.)
                    let n = probe(p);
                    println!("INFO record: {} — {n:?} frames", p.display());
                    if !n.is_some_and(|n| n >= min) {
                        fails.push(format!("{} reads back {n:?} frames", p.display()));
                    }
                }
                if cfg!(target_os = "macos") && s.mp4.is_none() {
                    fails.push("no MP4 (the OS encoder)".into());
                }
                if s.gif.is_none() {
                    fails.push("no GIF".into());
                }
            }
            other => fails.push(format!("save: {other:?}")),
        }
        let _ = std::fs::remove_dir_all(&dir);
        if fails.is_empty() {
            println!("TEST PASS record: a take of the window while playing, {} frames in review, saved to MP4 + GIF and read back", frames.unwrap_or(0));
        } else {
            println!("TEST FAIL record: {}", fails.join(" | "));
        }
    }

    /// Colour management against wobble: an EXR of flat scene-linear
    /// patches (`testdata/colour/patches.1001.exr`, some past 1.0) in an
    /// ACEScg project through the ACES 1.3 view must show what wobble's
    /// viewer shows for it (`testdata/colour/wobble_aces13.png`, wobble's
    /// `viewer_capture` with default prefs: ACEScg, ACES 1.3, sRGB), patch
    /// by patch; then exposure +1 brightens mid grey and a Cineon input on
    /// the same file changes the picture (the input transform runs).
    fn test_colour(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/colour");
        let (exr, refp) = (dir.join("patches.1001.exr"), dir.join("wobble_aces13.png"));
        let Ok(reference) = decode_png(&refp) else {
            println!("TEST FAIL colour: no {} (make it with wobble's viewer_capture)", refp.display());
            return;
        };
        let ed = &mut self.editor;
        ed.open(&exr);
        ed.colour_mut().working = colour::color::Colorspace::ACESCG;
        ed.colour_mut().view.transform = colour::viewtf::ViewTransform::Aces13;
        ed.wait_for_picture(std::time::Duration::from_secs(10));
        let mut fails = Vec::new();
        let mut shot = |ed: &mut editor::Editor| -> Option<(Vec<u8>, u32, egui::Rect, f32)> {
            let mut tex = None;
            for _ in 0..4 {
                tex = Some(win.offscreen(gpu, ed));
            }
            let tex = tex?;
            let px = imfram::shell::capture::read_pixels(gpu, &tex)?;
            let bgra = matches!(tex.format(), wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
            let px = if bgra { px.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0], p[3]]).collect() } else { px };
            Some((px, tex.width(), ed.picture_rect(0)?, win.ctx().pixels_per_point()))
        };
        let at = |s: &(Vec<u8>, u32, egui::Rect, f32), u: f32, v: f32| -> [u8; 3] {
            let (px, w, r, ppp) = s;
            let x = ((r.min.x + u * r.width()) * ppp) as usize;
            let y = ((r.min.y + v * r.height()) * ppp) as usize;
            let o = (y * *w as usize + x) * 4;
            [px[o], px[o + 1], px[o + 2]]
        };
        let Some(s) = shot(ed) else {
            println!("TEST FAIL colour: no picture");
            return;
        };
        let mut worst = 0i32;
        for j in 0..4 {
            for i in 0..6 {
                let (u, v) = ((i as f32 + 0.5) / 6.0, (j as f32 + 0.5) / 4.0);
                let e = at(&s, u, v);
                let (rw, rh) = (reference.1 as f32, reference.2 as f32);
                let o = (((v * rh) as usize) * reference.1 as usize + (u * rw) as usize) * 3;
                let w = [reference.0[o], reference.0[o + 1], reference.0[o + 2]];
                let d = (0..3).map(|k| (e[k] as i32 - w[k] as i32).abs()).max().unwrap_or(0);
                worst = worst.max(d);
                if d > 2 {
                    fails.push(format!("patch {i},{j}: scrub {e:?} wobble {w:?}"));
                }
            }
        }
        println!("INFO colour: 24 patches through ACES 1.3 against wobble's viewer, worst difference {worst}/255");
        // Exposure +1 stop brightens mid grey (patch 3,0: 0.18).
        let grey = at(&s, 3.5 / 6.0, 0.5 / 4.0);
        ed.colour_mut().view.exposure = 1.0;
        let s2 = shot(ed).expect("picture");
        let grey2 = at(&s2, 3.5 / 6.0, 0.5 / 4.0);
        if grey2[1] <= grey[1] + 10 {
            fails.push(format!("exposure +1: mid grey {grey:?} -> {grey2:?}"));
        }
        ed.colour_mut().view.exposure = 0.0;
        // The export path is the same passes: a still of the timeline shows
        // the viewer's patch values (mid grey, 1.0, a saturated red).
        ed.build_cuts(1, false);
        let seq = ed.sequence().clone();
        let read: Vec<std::path::PathBuf> = seq.media.iter().map(|m| m.path.clone()).collect();
        match crate::render::still(&seq, &read, 0, 1920) {
            Ok((bytes, w, h)) => {
                let png_path = std::env::temp_dir().join(format!("scrub-colour-still-{}.png", std::process::id()));
                let _ = std::fs::write(&png_path, &bytes);
                if let Ok((rgb, _, _)) = decode_png(&png_path) {
                    let mut worst = 0;
                    for (i, j) in [(3usize, 0usize), (5, 0), (2, 1), (2, 3)] {
                        let (u, v) = ((i as f32 + 0.5) / 6.0, (j as f32 + 0.5) / 4.0);
                        let o = (((v * h as f32) as usize) * w as usize + (u * w as f32) as usize) * 3;
                        let ex = [rgb[o], rgb[o + 1], rgb[o + 2]];
                        let vi = at(&s, u, v);
                        // (4:2:0 + limited range on the way to H.264: a few codes.)
                        let d = (0..3).map(|k| (ex[k] as i32 - vi[k] as i32).abs()).max().unwrap_or(0);
                        worst = worst.max(d);
                        if d > 6 {
                            fails.push(format!("export still patch {i},{j}: {ex:?} viewer {vi:?}"));
                        }
                    }
                    println!("INFO colour: export still against the viewer, worst {worst}/255 (through 4:2:0)");
                }
                let _ = std::fs::remove_file(&png_path);
            }
            Err(e) => fails.push(format!("export still: {e:#}")),
        }
        // Read as Cineon instead: the input transform changes the picture.
        ed.colour_mut().inputs.insert(exr.clone(), colour::color::InputSpace::Explicit(colour::color::Colorspace::CINEON));
        let s3 = shot(ed).expect("picture");
        let g3 = at(&s3, 3.5 / 6.0, 0.5 / 4.0);
        if (g3[1] as i32 - grey[1] as i32).abs() < 10 {
            fails.push(format!("a Cineon input left mid grey as it was: {grey:?} -> {g3:?}"));
        }
        if fails.is_empty() {
            println!("TEST PASS colour: an ACEScg EXR through ACES 1.3 matches wobble's viewer patch by patch (worst {worst}/255), exposure, input transform");
        } else {
            println!("TEST FAIL colour: {}", fails.join(" | "));
        }
    }

    /// The project file end to end: cut a sequence with speed changes and
    /// dissolves, save it, open it in a fresh editor, compare everything;
    /// then move the media and check it is found by its relative path.
    fn test_project(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let Some(src) = self.editor.current().map(|(p, _, _)| p.to_path_buf()) else {
            println!("TEST FAIL no clip");
            return;
        };
        let dir = std::env::temp_dir().join(format!("scrub-test-project-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("media")).expect("temp dir");
        let media = dir.join("media").join(src.file_name().expect("a file"));
        std::fs::copy(&src, &media).expect("copy the clip");
        let mut fails = Vec::new();
        let mut a = editor::Editor::new(true);
        a.open(&media);
        a.build_cuts(12, true);
        let _ = win.offscreen(gpu, &mut a);
        let p = dir.join("cut.scrub");
        if let Err(e) = a.save_to(&p) {
            fails.push(format!("save: {e}"));
        }
        let mut b = editor::Editor::new(true);
        b.open_project(&p);
        let _ = win.offscreen(gpu, &mut b);
        let (sa, sb) = (a.sequence_summary(), b.sequence_summary());
        if sa.clips != sb.clips || sa.dissolves != sb.dissolves || sa.speeds != sb.speeds {
            fails.push(format!("round trip differs: {sa:?} vs {sb:?}"));
        }
        if a.project_doc(Some(&p)).sequence != b.project_doc(Some(&p)).sequence {
            fails.push("the sequences differ".into());
        }
        // Move the whole folder: found again by the relative path. Close
        // both editors first: Windows will not move a folder with files
        // open in it (their decoders let go on their own threads, so retry).
        drop((a, b));
        let moved = dir.with_extension("moved");
        let _ = std::fs::remove_dir_all(&moved);
        let mut tries = 0;
        while let Err(e) = std::fs::rename(&dir, &moved) {
            tries += 1;
            assert!(tries < 50, "move: {e:?}");
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let mut c = editor::Editor::new(true);
        c.open_project(&moved.join("cut.scrub"));
        if !c.project.missing.is_empty() {
            fails.push(format!("{} media offline after the move", c.project.missing.len()));
        }
        let _ = win.offscreen(gpu, &mut c);
        c.wait_for_picture(std::time::Duration::from_secs(5));
        let _ = std::fs::remove_dir_all(&moved);
        if fails.is_empty() {
            println!("TEST PASS project save, reopen and relative relink");
        } else {
            println!("TEST FAIL {}", fails.join(" | "));
        }
    }

    /// Three-point editing through the keyboard, as a user would: mark the
    /// source, overwrite, insert, undo, redo, walk the edit points, patch a
    /// target. Prints one PASS / FAIL line.
    fn test_keys(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let press = |win: &mut UiWindow, ed: &mut editor::Editor, key: egui::Key, mods: egui::Modifiers| {
            for pressed in [true, false] {
                win.inject([egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: mods }]);
                let _ = win.offscreen(gpu, ed);
            }
        };
        use egui::{Key as K, Modifiers as M};
        let ed = &mut self.editor;
        let _ = win.offscreen(gpu, ed);
        let mut fails = Vec::new();
        let mut check = |what: &str, ok: bool| {
            if !ok {
                fails.push(what.to_string());
            }
        };
        press(win, ed, K::Num2, M::SHIFT); // focus source
        if ed.targets_audio() {
            press(win, ed, K::Num4, M::NONE); // A1 off: the checks below count picture clips
        }
        for _ in 0..10 {
            press(win, ed, K::ArrowRight, M::NONE);
        }
        press(win, ed, K::I, M::NONE);
        for _ in 0..9 {
            press(win, ed, K::ArrowRight, M::NONE);
        }
        press(win, ed, K::O, M::NONE);
        press(win, ed, K::Period, M::NONE); // overwrite at record 0
        let s = ed.sequence_summary();
        check(&format!("overwrite: {s:?}"), s.duration == 10 && s.clips == vec![(0, 0, 10, 10)]);
        press(win, ed, K::Num2, M::NONE); // target V2
        press(win, ed, K::Comma, M::NONE); // insert at the record playhead (10)
        let s = ed.sequence_summary();
        check(&format!("insert: {s:?}"), s.duration == 20 && s.clips.contains(&(1, 10, 10, 10)) && s.playhead == 20);
        press(win, ed, K::ArrowUp, M::NONE); // to the edit point at 10
        check(&format!("up: {:?}", ed.sequence_summary().playhead), ed.sequence_summary().playhead == 10);
        press(win, ed, K::Z, M::COMMAND);
        check("undo", ed.sequence_summary().duration == 10);
        press(win, ed, K::Z, M::COMMAND | M::SHIFT);
        check("redo", ed.sequence_summary().duration == 20);
        // Record marks back-time: an out at 4 with no in ends the edit there.
        press(win, ed, K::Num3, M::SHIFT);
        press(win, ed, K::Home, M::NONE);
        for _ in 0..4 {
            press(win, ed, K::ArrowRight, M::NONE);
        }
        press(win, ed, K::O, M::NONE);
        press(win, ed, K::Num3, M::NONE); // target V3
        press(win, ed, K::Period, M::NONE);
        let s = ed.sequence_summary();
        check(&format!("back-timed overwrite on V3: {s:?}"), s.clips.contains(&(2, 0, 5, 15)) && s.playhead == 5);

        // Step 3. Select the V3 clip under the playhead and slip it.
        press(win, ed, K::Home, M::NONE);
        press(win, ed, K::D, M::NONE);
        press(win, ed, K::Period, M::ALT);
        press(win, ed, K::Period, M::ALT);
        check(&format!("slip: {:?}", ed.sequence_summary().clips), ed.sequence_summary().clips.contains(&(2, 0, 5, 17)));
        press(win, ed, K::Escape, M::NONE);
        // A second clip on V1 at 10, then roll the cut between them.
        press(win, ed, K::Num1, M::NONE);
        press(win, ed, K::Home, M::NONE);
        for _ in 0..10 {
            press(win, ed, K::ArrowRight, M::NONE);
        }
        press(win, ed, K::Period, M::NONE);
        press(win, ed, K::Home, M::NONE);
        for _ in 0..11 {
            press(win, ed, K::ArrowRight, M::NONE);
        }
        press(win, ed, K::G, M::NONE); // the cut at 10, roll
        press(win, ed, K::ArrowLeft, M::ALT);
        press(win, ed, K::ArrowLeft, M::ALT);
        let s = ed.sequence_summary();
        check(&format!("roll -2: {:?}", s.clips), s.clips.contains(&(0, 0, 8, 10)) && s.clips.contains(&(0, 8, 12, 8)));
        // T again: the out side; in ripple mode (B) a trim +1 moves what
        // follows.
        press(win, ed, K::G, M::NONE);
        press(win, ed, K::B, M::NONE);
        press(win, ed, K::ArrowRight, M::ALT);
        press(win, ed, K::B, M::NONE);
        let s = ed.sequence_summary();
        check(&format!("ripple out +1: {:?}", s.clips), s.clips.contains(&(0, 0, 9, 10)) && s.clips.contains(&(0, 9, 12, 8)));
        // A dissolve on that cut (clamped: the incoming clip has 8 frames of head).
        press(win, ed, K::D, M::COMMAND);
        let s = ed.sequence_summary();
        check(&format!("dissolve: {:?}", s.dissolves), s.dissolves.iter().any(|d| d.0 == 0 && d.1 == 9 && d.2 == 8));
        // Extend: roll the cut to the playhead.
        press(win, ed, K::G, M::NONE);
        press(win, ed, K::G, M::NONE); // back to roll
        press(win, ed, K::Home, M::NONE);
        for _ in 0..6 {
            press(win, ed, K::ArrowRight, M::NONE);
        }
        press(win, ed, K::E, M::NONE);
        let s = ed.sequence_summary();
        check(&format!("extend to 6: {:?}", s.clips), s.clips.iter().any(|c| c.0 == 0 && c.1 == 6));
        // Speed: cmd+R opens the dialog; apply 2x to the clip at the playhead.
        press(win, ed, K::Escape, M::NONE);
        press(win, ed, K::D, M::NONE);
        press(win, ed, K::R, M::COMMAND);
        check("speed dialog opens", ed.speed_dialog.is_some());
        press(win, ed, K::Escape, M::NONE);
        ed.apply_speed(2.0);
        let s = ed.sequence_summary();
        check(&format!("speed 2x: {:?}", s.speeds), s.speeds.iter().any(|x| x.2 == 2.0));
        // Reframe the selected clip: fill, zoom in a step, move right twice.
        press(win, ed, K::F, M::NONE);
        press(win, ed, K::CloseBracket, M::NONE);
        press(win, ed, K::ArrowRight, M::COMMAND);
        press(win, ed, K::ArrowRight, M::COMMAND);
        let line = ed.reframe_line().unwrap_or_default();
        check(&format!("reframe keys: {line}"), line == "reframe: fill 105%  x +2%  y +0%");
        press(win, ed, K::F, M::ALT);
        check("alt+F resets the reframe", ed.reframe_line().is_none());
        // Delete lifts the selected clip.
        let before = ed.sequence_summary().clips.len();
        press(win, ed, K::Delete, M::NONE);
        check("delete lifts a clip", ed.sequence_summary().clips.len() + 1 == before);
        if fails.is_empty() {
            println!("TEST PASS three-point editing and step 3 edits through the keys");
        } else {
            println!("TEST FAIL {}", fails.join(" | "));
        }
    }

    /// The in-app optimise end to end, to the managed cache: the job runs
    /// in the background while redraws continue, and when it lands the
    /// clip reads the optimised copy at the same frame.
    fn test_optimise(&mut self) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let Some((path, opt, _)) = self.editor.current().map(|(p, o, i)| (p.to_path_buf(), o, i)) else {
            println!("TEST FAIL no clip");
            return;
        };
        if opt {
            println!("TEST FAIL already optimised (purge first)");
            return;
        }
        self.editor.seek_display(57);
        let _ = win.offscreen(gpu, &mut self.editor);
        self.editor.opt.start(&path, media::optimise::Dest::Cache, media::optimise::Profile::Standard);
        let t0 = std::time::Instant::now();
        let mut redraws = 0;
        while t0.elapsed().as_secs() < 120 {
            let _ = win.offscreen(gpu, &mut self.editor);
            redraws += 1;
            if !self.editor.opt.busy() {
                break;
            }
        }
        let _ = win.offscreen(gpu, &mut self.editor);
        let now = self.editor.current().map(|(_, o, _)| o).unwrap_or(false);
        let frame = self.editor.frame_display();
        let ok = now && frame == Some(57);
        println!(
            "TEST {} optimised={now} frame={frame:?} after {:.1}s, {redraws} redraws while it ran",
            if ok { "PASS" } else { "FAIL" },
            t0.elapsed().as_secs_f32()
        );
    }

    /// Headless playback: every redraw is a real one (pick, upload, GPU
    /// convert, draw, wait for the GPU), paced like a 60 Hz display.
    fn play(&mut self, secs: f64, speed: f64) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        self.editor.wait_for_picture(std::time::Duration::from_secs(10));
        self.editor.play(speed);
        let p0 = self.editor.playhead_pos();
        let t0 = std::time::Instant::now();
        let mut steady: Option<(f64, std::time::Instant)> = None;
        let (mut redraws, mut upload, mut slow) = (0u64, 0f64, 0u64);
        let base = self.editor.shown_total();
        let mut last_drop = 0;
        while t0.elapsed().as_secs_f64() < secs {
            let due = t0 + std::time::Duration::from_secs_f64((redraws + 1) as f64 / 60.0);
            let r0 = std::time::Instant::now();
            let _ = win.offscreen(gpu, &mut self.editor);
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            if r0.elapsed().as_secs_f64() > 1.0 / 60.0 {
                slow += 1;
                if media::sys::var_os("TRACE").is_some() {
                    eprintln!("slow redraw {redraws}: {:.1} ms at frame {:?}, dropped so far {}", r0.elapsed().as_secs_f64() * 1000.0, self.editor.frame_display(), self.editor.dropped());
                }
            }
            if let Some(v) = win.layer_mut().renderer_mut().callback_resources.get::<viewer::Viewers>() {
                upload += v.slots.iter().map(|s| s.upload_ms as f64).fold(0.0, f64::max);
            }
            if media::sys::var_os("TRACE").is_some() && self.editor.dropped() != last_drop {
                last_drop = self.editor.dropped();
                eprintln!("drop at redraw {redraws}, playhead {:?}, {:.1} ms in", self.editor.frame_display(), t0.elapsed().as_secs_f64() * 1000.0);
            }
            redraws += 1;
            if steady.is_none() && t0.elapsed().as_secs_f64() >= 1.5 {
                steady = Some((self.editor.playhead_pos(), std::time::Instant::now()));
            }
            std::thread::sleep(due.saturating_duration_since(std::time::Instant::now()));
        }
        let el = t0.elapsed().as_secs_f64();
        if let Some((ps, ts)) = steady {
            let (dt, fps) = (ts.elapsed().as_secs_f64(), self.editor.active_fps());
            println!("steady: playhead {:.3}x real time after 1.5 s", (self.editor.playhead_pos() - ps) / (dt * fps).max(1e-9));
        }
        let (p1, fps) = (self.editor.playhead_pos(), self.editor.active_fps());
        println!("realtime: playhead {:.1} frames in {el:.2}s at {fps} fps = {:.3}x", p1 - p0, (p1 - p0) / (el * fps).max(1e-9));
        println!("{}", self.editor.audio_status());
        let shown = self.editor.shown_total() - base;
        println!(
            "play {speed}x for {el:.1}s: {:.1} pictures/s, {} dropped, {redraws} redraws ({slow} over 16.7 ms), upload {:.1} ms mean",
            shown as f64 / el,
            self.editor.dropped(),
            upload / redraws.max(1) as f64
        );
    }

    fn shoot(&mut self, path: &std::path::Path) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        self.editor.wait_for_picture(std::time::Duration::from_secs(10));
        // A few passes: egui sizes some things from the pass before.
        let mut tex = None;
        for _ in 0..4 {
            tex = Some(win.offscreen(gpu, &mut self.editor));
        }
        imfram::shell::capture::save_png(gpu, &tex.expect("drawn"), path).expect("saving the shot");
        eprintln!("wrote {}", path.display());
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.win.is_some() {
            return;
        }
        // EXPERIMENT (fold): the device is made the way wobble needs it, so
        // the hosted wobble (started on demand) shares it.
        #[cfg(feature = "fold")]
        let gpu = {
            let seed = wobble_app::hosted::Seed::new().expect("no GPU adapter");
            let (i, a, d, q) = seed.handles();
            self.editor.fold.seed = Some(seed);
            Gpu::new(i, a, d, q)
        };
        #[cfg(not(feature = "fold"))]
        let gpu = Gpu::standalone().expect("no GPU adapter");
        let headless = self.shot.is_some() || self.play.is_some() || self.test_optimise || self.test_keys || self.test_mouse || self.test_tools || self.test_api || self.test_wobble || self.test_project || self.test_colour || self.test_record || self.test_realtime;
        let mut win = UiWindow::new(el, &gpu, media::sys::APP, (1600, 1000), !headless);
        if !headless {
            appicon::apply(&win.window);
        }
        let res = viewer::Viewers::new(&gpu.device, win.format(), 3);
        win.layer_mut().renderer_mut().callback_resources.insert(res);
        #[cfg(feature = "fold")]
        {
            let blit = viewer::TexBlit::new(&gpu.device, win.format());
            win.layer_mut().renderer_mut().callback_resources.insert(blit);
        }
        editor::style(win.ctx());
        self.editor.attach_window(win.window.clone(), win.ctx().clone(), headless);
        self.editor.device = Some(gpu.device.clone());
        self.gpu = Some(gpu);
        self.win = Some(win);
        if self.test_optimise {
            self.test_optimise();
        }
        if self.test_keys {
            self.test_keys();
        }
        if self.test_colour {
            self.test_colour();
        }
        if self.test_record {
            self.test_record();
        }
        if self.test_realtime {
            self.test_realtime();
        }
        if self.test_project {
            self.test_project();
        }
        if self.test_mouse {
            self.test_mouse(el);
        }
        if self.test_tools {
            self.test_tools();
        }
        if self.test_api {
            self.test_api();
        }
        #[cfg(feature = "wobble")]
        if self.test_wobble {
            self.test_wobble();
        }
        if let Some((secs, speed)) = self.play {
            self.play(secs, speed);
        }
        // EDIFY_SHOT_CLIPS=<folder>: the shot with the clip browser open there.
        #[cfg(feature = "clips")]
        if let Some(d) = media::sys::var_os("SHOT_CLIPS").filter(|_| self.shot.is_some()) {
            self.editor.import();
            self.editor.clips_go(std::path::Path::new(&d));
        }
        // EDIFY_SHOT_TILES=<folder>: the shot with that folder's clips in the
        // bin as tiles (the posters given time to land).
        if let Some(d) = media::sys::var_os("SHOT_TILES").filter(|_| self.shot.is_some()) {
            let mut v: Vec<_> = std::fs::read_dir(&d).map(|r| r.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "mov" || x == "mp4")).collect()).unwrap_or_default();
            v.sort();
            for p in v {
                self.editor.bin_add(&p);
            }
            self.editor.bin_set_tiles(true);
            if let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) {
                for _ in 0..200 {
                    let _ = win.offscreen(gpu, &mut self.editor);
                    std::thread::sleep(std::time::Duration::from_millis(15));
                }
            }
        }
        #[cfg(feature = "fold")]
        if let Some(d) = std::env::var_os("SCRUB_TEST_FOLD") {
            self.test_fold(el, std::path::Path::new(&d));
        }
        if let Some(p) = self.shot.clone() {
            self.shoot(&p);
        }
        if headless {
            // Leave at once: an event loop asked to exit from inside a long
            // blocking `resumed` with a CoreAudio stream open did not.
            self.editor.shutdown();
            el.exit();
            std::process::exit(0);
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        // A torn-off panel's window: the dock's.
        if id != win.id() {
            let _ours = self.editor.dock_event(gpu, id, &event);
            #[cfg(feature = "fold")]
            if !_ours {
                self.editor.fold_window_event(el, id, event);
            }
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                if self.editor.request_quit() {
                    self.editor.shutdown();
                    el.exit();
                } else {
                    win.request_repaint();
                }
                return;
            }
            WindowEvent::RedrawRequested => {
                // The recorder copies the presented frame (UI and all); the
                // surface is copyable only while it wants one.
                let (rec, shot) = (self.editor.recorder.clone(), self.editor.shot.clone());
                let want = rec.lock().is_ok_and(|r| r.recording()) || shot.lock().is_ok_and(|s| s.wants_copy());
                win.set_copy_src(gpu, want);
                let ppp = win.ctx().pixels_per_point();
                let cursor = win.ctx().input(|i| i.pointer.hover_pos()).map(|p| [p.x * ppp, p.y * ppp]);
                let fmt = win.format();
                win.draw_with(gpu, &mut self.editor, &mut |tex| {
                    if !want {
                        return;
                    }
                    let mut enc = gpu.device.create_command_encoder(&Default::default());
                    if let Ok(mut r) = rec.lock() {
                        r.capture(&mut enc, tex, fmt, cursor);
                    }
                    if let Ok(mut s) = shot.lock() {
                        if s.wants_copy() {
                            s.capture(&gpu.device, &mut enc, tex, fmt);
                        }
                    }
                    gpu.queue.submit([enc.finish()]);
                    if let Ok(mut r) = rec.lock() {
                        r.after_submit();
                    }
                    if let Ok(mut s) = shot.lock() {
                        s.after_submit();
                    }
                });
                self.editor.dock_repaint();
                let title = self.editor.title();
                if title != self.title {
                    win.window.set_title(&title);
                    self.title = title;
                }
                if let Some(v) = win.layer_mut().renderer_mut().callback_resources.get::<viewer::Viewers>() {
                    for (i, s) in v.slots.iter().enumerate().take(2) {
                        self.editor.upload_ms[i] = s.upload_ms;
                    }
                }
                if self.editor.quit {
                    self.editor.shutdown();
                    el.exit();
                }
                return;
            }
            _ => {}
        }
        win.on_event(gpu, &event);
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        if let Some(gpu) = &self.gpu {
            self.editor.dock_between(el, gpu);
        }
        #[cfg(feature = "fold")]
        self.editor.fold_between(el);
        if let Some(win) = &mut self.win {
            win.poll();
        }
        el.set_control_flow(ControlFlow::Wait);
    }
}

fn main() {
    appicon::drop_lonely_console();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let mut args = std::env::args().skip(1).peekable();
    let (mut path, mut shot, mut frame, mut play, mut speed) = (None, None, None, None, 1.0);
    let (mut optimise, mut dest, mut profile, mut test_opt) = (false, None, None, false);
    let mut cuts = None;
    let mut fx = false;
    let mut subs: Option<usize> = None;
    let mut test_keys = false;
    let mut test_project = false;
    let mut test_colour = false;
    let mut test_record = false;
    let mut test_realtime = false;
    let mut test_mouse = false;
    let mut test_tools = false;
    let mut test_api = false;
    let mut test_wobble = false;
    // The scripting API: on for an interactive window unless --no-api;
    // `--api PORT` picks the port (default 7980 or EDIFY_API_PORT).
    let mut api_port: Option<u16> = None;
    let mut no_api = false;
    let mut test_render = false;
    let mut test_post = false;
    let mut render: Option<(String, PathBuf)> = None;
    let mut fill = false;
    let mut reframe_test = false;
    let mut post: Option<PathBuf> = None;
    let mut rushes: Vec<PathBuf> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--shot" => shot = args.next().map(PathBuf::from),
            "--play" => play = args.next().and_then(|s| s.parse::<f64>().ok()),
            "--speed" => speed = args.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0),
            "--optimise" => optimise = true,
            "--fx" => fx = true,
            "--subs" => subs = args.next().and_then(|s| s.parse::<usize>().ok()),
            "--cuts" => cuts = args.next().and_then(|s| s.parse::<usize>().ok()),
            "--test-optimise" => test_opt = true,
            "--test-keys" => test_keys = true,
            "--test-project" => test_project = true,
            "--test-colour" => test_colour = true,
            "--test-record" => test_record = true,
            "--test-realtime" => test_realtime = true,
            "--test-mouse" => test_mouse = true,
            "--test-tools" => test_tools = true,
            "--test-api" => test_api = true,
            "--test-wobble" => test_wobble = true,
            "--no-api" => no_api = true,
            "--api" => {
                api_port = Some(match args.peek().and_then(|s| s.parse::<u16>().ok()) {
                    Some(p) => {
                        args.next();
                        p
                    }
                    None => api::port_from_env(),
                })
            }
            "--test-render" => test_render = true,
            "--test-post" => test_post = true,
            "--render" => {
                let preset = args.next().unwrap_or_default();
                render = args.next().map(|o| (preset, PathBuf::from(o)));
            }
            "--fill" => fill = true,
            "--reframe-test" => reframe_test = true,
            "--post" => post = args.next().map(PathBuf::from),
            "--rushes" => rushes.extend(args.next().map(PathBuf::from)),
            "--dest" => dest = args.next().and_then(|s| media::optimise::Dest::parse(&s)),
            "--profile" => profile = args.next().and_then(|s| media::optimise::Profile::parse(&s)),
            "--frame" => frame = args.next().and_then(|s| s.parse::<i64>().ok()),
            _ => path = Some(PathBuf::from(a)),
        }
    }
    if optimise {
        let Some(p) = path else {
            eprintln!("--optimise needs a file");
            std::process::exit(2);
        };
        std::process::exit(optimise_cli(&p, dest, profile));
    }
    let headless = shot.is_some() || play.is_some() || test_opt || test_keys || test_mouse || test_tools || test_api || test_wobble || test_project || test_colour || test_record || test_realtime || test_render || test_post || render.is_some();
    let mut editor = editor::Editor::new(headless);
    if let Some(p) = path.take_if(|p| p.extension().is_some_and(|e| e == project::EXT || e == project::OLD_EXT)) {
        editor.open_project(&p);
    }
    // A post folder or a post.json given plainly is a post too.
    if post.is_none() {
        post = path.take_if(|p| p.file_name().is_some_and(|n| n == marketing::POST_FILE) || p.join(marketing::POST_FILE).is_file());
    }
    if let Some(p) = &post {
        if let Err(e) = editor.open_post_path(p, &rushes) {
            eprintln!("{e}");
            editor.notice(e);
        }
    }
    if let Some(p) = path {
        editor.open(&p);
        if let Some(n) = cuts {
            editor.build_cuts(n, fx);
            editor.focus_record();
        }
        if let Some(n) = subs {
            editor.build_subs(n);
        }
        if reframe_test {
            // A 9:16 sequence with every clip filled and moved 20% right.
            editor.reframe_all(Some((1080, 1920)), model::Reframe { fill: true, zoom: 1.0, x: 0.2, y: 0.0 });
        }
        if let Some(f) = frame {
            editor.seek_display(f);
        }
    }
    if let Some((preset, out)) = render {
        std::process::exit(render_cli(&mut editor, &preset, &out, fill));
    }
    if test_render {
        std::process::exit(test_render_cli(&mut editor));
    }
    if test_post {
        std::process::exit(test_post_cli(&mut editor));
    }
    if !no_api && (!headless || api_port.is_some()) && !test_api {
        let port = api_port.unwrap_or_else(api::port_from_env);
        match editor.api_listen(port) {
            Ok(p) => log::info!("scripting API on 127.0.0.1:{p} (scrub-mcp bridges it to MCP)"),
            Err(e) => {
                eprintln!("{e}");
                editor.notice(e);
            }
        }
    }
    let el = EventLoop::new().expect("event loop");
    let mut app = App { gpu: None, win: None, editor, shot, test_optimise: test_opt, test_keys, test_mouse, test_tools, test_api, test_wobble, test_project, test_colour, test_record, test_realtime, title: String::new(), play: play.map(|p| (p, speed)) };
    el.run_app(&mut app).expect("run");
}

/// `--optimise`: transcode with progress on stderr, exit status for scripts.
fn optimise_cli(path: &std::path::Path, dest: Option<media::optimise::Dest>, profile: Option<media::optimise::Profile>) -> i32 {
    use media::optimise::{Job, Progress, Settings};
    use std::sync::atomic::Ordering;
    let s = Settings::load();
    let job = Job {
        source: path.to_path_buf(),
        dest: dest.unwrap_or(s.dest),
        profile: profile.unwrap_or(s.profile),
        cache_dir: s.cache_dir.clone(),
        cache_cap_gb: s.cache_cap_gb,
    };
    let progress = std::sync::Arc::new(Progress::default());
    let p = progress.clone();
    let t0 = std::time::Instant::now();
    let h = std::thread::spawn(move || media::optimise::run(&job, &p));
    while !h.is_finished() {
        std::thread::sleep(std::time::Duration::from_millis(500));
        eprint!("\r{:5.1}%  {} / {}", progress.fraction() * 100.0, progress.done.load(Ordering::Relaxed), progress.total.load(Ordering::Relaxed));
    }
    eprintln!();
    match h.join() {
        Ok(Ok(out)) => {
            let n = progress.done.load(Ordering::Relaxed);
            let secs = t0.elapsed().as_secs_f64();
            println!("optimised {n} frames in {secs:.1}s ({:.0} fps) -> {}", n as f64 / secs, out.display());
            0
        }
        Ok(Err(e)) => {
            eprintln!("optimise failed: {e:#}");
            1
        }
        Err(_) => 1,
    }
}

/// `--render PRESET OUT.mp4 [--fill]`: render the record sequence, no window.
fn render_cli(ed: &mut editor::Editor, preset: &str, out: &std::path::Path, fill: bool) -> i32 {
    let Some(p) = render::preset(preset) else {
        eprintln!("unknown preset {preset}; one of: {}", render::PRESETS.iter().map(|p| p.id).collect::<Vec<_>>().join(", "));
        return 2;
    };
    let fit = if fill { render::Fit::Fill } else { render::Fit::Fit };
    let job = ed.render_job(p, fit, false, out.to_path_buf(), None);
    let progress = render::Progress::default();
    match render::run(&job, &progress) {
        Ok(d) => {
            println!(
                "rendered {} frames in {:.2}s: {:.1} fps with {} -> {}",
                d.frames,
                d.secs,
                d.frames as f64 / d.secs,
                d.encoder,
                d.out.display()
            );
            0
        }
        Err(e) => {
            eprintln!("render failed: {e:#}");
            1
        }
    }
}

/// `--test-render`: cut the clip into a sequence with speed changes and
/// dissolves, render it for YouTube (fit) and Reels (fill), and check both
/// files with ffprobe: codecs, size, frame count, duration, sound.
fn test_render_cli(ed: &mut editor::Editor) -> i32 {
    ed.build_cuts(10, true);
    let dir = std::env::temp_dir().join(format!("scrub-test-render-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let mut fails = Vec::new();
    let summary = ed.sequence_summary();
    let frames = summary.duration;
    let has_audio = ed.sequence_has_audio();
    for (preset, fit) in [("youtube-1080", render::Fit::Fit), ("reels", render::Fit::Fill)] {
        let p = render::preset(preset).expect("preset");
        let out = dir.join(format!("{preset}.mp4"));
        let job = ed.render_job(p, fit, false, out.clone(), None);
        let progress = render::Progress::default();
        let d = match render::run(&job, &progress) {
            Ok(d) => d,
            Err(e) => {
                fails.push(format!("{preset}: {e:#}"));
                continue;
            }
        };
        println!("{preset}: {} frames in {:.2}s ({:.1} fps, {})", d.frames, d.secs, d.frames as f64 / d.secs, d.encoder);
        let probe = std::process::Command::new("ffprobe")
            .args(["-v", "error", "-count_frames", "-show_entries", "stream=codec_type,codec_name,width,height,nb_read_frames:format=duration", "-of", "json"])
            .arg(&d.out)
            .output();
        let Ok(probe) = probe else {
            fails.push("no ffprobe".into());
            continue;
        };
        let v: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap_or_default();
        let streams = v["streams"].as_array().cloned().unwrap_or_default();
        let video = streams.iter().find(|s| s["codec_type"] == "video");
        let audio = streams.iter().find(|s| s["codec_type"] == "audio");
        match video {
            Some(s) => {
                if s["codec_name"] != "h264" || s["width"] != p.width || s["height"] != p.height {
                    fails.push(format!("{preset}: video {s}"));
                }
                let n: i64 = s["nb_read_frames"].as_str().and_then(|x| x.parse().ok()).unwrap_or(-1);
                if n != frames {
                    fails.push(format!("{preset}: {n} frames, the sequence has {frames}"));
                }
            }
            None => fails.push(format!("{preset}: no video stream")),
        }
        if has_audio && !audio.is_some_and(|a| a["codec_name"] == "aac") {
            fails.push(format!("{preset}: no AAC sound"));
        }
        let dur: f64 = v["format"]["duration"].as_str().and_then(|x| x.parse().ok()).unwrap_or(0.0);
        let want = frames as f64 / ed.sequence_fps();
        if (dur - want).abs() > 0.1 {
            fails.push(format!("{preset}: {dur:.3}s long, the sequence is {want:.3}s"));
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    if fails.is_empty() {
        println!("TEST PASS render: YouTube fit and Reels fill, checked with ffprobe");
        0
    } else {
        println!("TEST FAIL {}", fails.join(" | "));
        1
    }
}

/// `--test-post FILE`: an organisicator marketing folder in a temp dir with
/// FILE as a rush and an advanced 9:16 post using it. Open the post, cut,
/// save, render: the project lands beside post.json, the render in out/ and
/// as the next result, the sequence is 1080x1920, and post.json is never
/// touched.
fn test_post_cli(ed: &mut editor::Editor) -> i32 {
    let Some(src) = ed.current().map(|(p, _, _)| p.to_path_buf()) else {
        println!("TEST FAIL no clip");
        return 1;
    };
    let root = std::env::temp_dir().join(format!("scrub-test-post-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let m = root.join("marketing");
    for d in ["rushes/2026-11-11", "posts/teaser/in", "posts/teaser/out", "posts/teaser/scripts", "result"] {
        std::fs::create_dir_all(m.join(d)).expect("fixture");
    }
    let rush = m.join("rushes/2026-11-11").join(src.file_name().expect("file"));
    std::fs::copy(&src, &rush).expect("copy the rush");
    let post_json = format!(r#"{{"title": "Teaser", "kind": "advanced", "created": 1, "frame": "9:16", "sources": ["rushes/2026-11-11/{}"]}}"#, rush.file_name().unwrap().to_string_lossy());
    std::fs::write(m.join("posts/teaser/post.json"), &post_json).expect("post.json");
    std::fs::write(m.join("result/teaser_v003.mp4"), b"older").expect("an older result");
    let mut fails = Vec::new();

    // A second day of rushes: another clip and a short image sequence.
    std::fs::create_dir_all(m.join("rushes/2026-11-12/seq")).expect("day 2");
    let other = m.join("rushes/2026-11-12").join(format!("b.{}", rush.extension().unwrap().to_string_lossy()));
    std::fs::copy(&src, &other).expect("copy");
    for f in 1..=3 {
        std::fs::write(m.join(format!("rushes/2026-11-12/seq/shot.{f:04}.png")), b"png").expect("seq");
    }
    // As organisicator launches it: the post folder and the rushes folders.
    if let Err(e) = ed.open_post_path(&m.join("posts/teaser"), &[m.join("rushes")]) {
        println!("TEST FAIL open_post_path: {e}");
        return 1;
    }
    let t0 = std::time::Instant::now();
    while ed.bin_scanning() && t0.elapsed().as_secs() < 10 {
        ed.bin_poll();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let bins = ed.bin_groups();
    let group_of = |p: &std::path::Path| bins.iter().find(|(_, x)| x == p).map(|(g, _)| g.clone());
    if group_of(&rush).as_deref() != Some("post: Teaser") {
        fails.push(format!("the post's rush is in {:?}, not the post's bin", group_of(&rush)));
    }
    if group_of(&other).as_deref() != Some("rushes/2026-11-12") {
        fails.push(format!("the other day's rush is in {:?}", group_of(&other)));
    }
    let seq_first = m.join("rushes/2026-11-12/seq/shot.0001.png");
    if group_of(&seq_first).as_deref() != Some("rushes/2026-11-12/seq") || bins.iter().filter(|(_, p)| p.starts_with(m.join("rushes/2026-11-12/seq"))).count() != 1 {
        fails.push(format!("the image sequence is not one item in its bin: {bins:?}"));
    }
    if ed.project_doc(None).sequence.raster != Some((1080, 1920)) {
        fails.push("the sequence is not 9:16".into());
    }

    ed.open(&rush);
    ed.build_cuts(6, false);
    let project = m.join("posts/teaser/teaser.scrub");
    ed.save(false);
    if !project.is_file() {
        fails.push("the project was not saved beside post.json".into());
    }
    let preset = render::preset("reels").expect("preset");
    let post_ctx = marketing::PostCtx::for_project(&project);
    match &post_ctx {
        None => fails.push("the saved project is not seen as the post's".into()),
        Some(p) => {
            let job = ed.render_job(preset, render::Fit::Fill, false, p.out_path(preset.id), None);
            match render::run(&job, &render::Progress::default()) {
                Ok(d) => {
                    if d.out != m.join("posts/teaser/out/teaser_reels_v001.mp4") {
                        fails.push(format!("render went to {}", d.out.display()));
                    }
                    if m.join("result/teaser_v004.mp4").exists() {
                        fails.push("Scrub wrote a result (publishing is organisicator's)".into());
                    }
                }
                Err(e) => fails.push(format!("render: {e:#}")),
            }
        }
    }
    if std::fs::read_to_string(m.join("posts/teaser/post.json")).ok().as_deref() != Some(post_json.as_str()) {
        fails.push("post.json was changed by opening, cutting or rendering".into());
    }
    // Setting the frame shape writes that one field.
    if let Err(e) = ed.set_post_frame("4:5") {
        fails.push(format!("set frame: {e}"));
    }
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(m.join("posts/teaser/post.json")).unwrap_or_default()).unwrap_or_default();
    if v["frame"] != "4:5" || v["title"] != "Teaser" || v["sources"].as_array().map(|a| a.len()) != Some(1) {
        fails.push(format!("post.json after setting the frame: {v}"));
    }
    if ed.project_doc(None).sequence.raster != Some((1080, 1350)) {
        fails.push("the raster did not follow the frame".into());
    }
    if std::fs::read_dir(m.join("posts/teaser/out")).map(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().starts_with('.'))).unwrap_or(false) {
        fails.push("a hidden partial file was left in out/".into());
    }
    if std::fs::read(m.join("result/teaser_v003.mp4")).ok().as_deref() != Some(b"older".as_slice()) {
        fails.push("an existing result was overwritten".into());
    }
    let _ = std::fs::remove_dir_all(&root);
    if fails.is_empty() {
        println!("TEST PASS marketing post: launched as orgo does, post's rushes in their bin and the rest by day, opened 9:16, saved beside post.json, rendered to out/ only, post.json untouched until its frame was set, then only that field");
        0
    } else {
        println!("TEST FAIL {}", fails.join(" | "));
        1
    }
}

/// An 8-bit RGB / RGBA PNG as (rgb, width, height).
fn decode_png(p: &std::path::Path) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(p)?));
    let mut r = dec.read_info()?;
    let mut buf = vec![0u8; r.output_buffer_size().unwrap_or(0)];
    let info = r.next_frame(&mut buf)?;
    let n = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        c => anyhow::bail!("{c:?} PNG"),
    };
    let rgb: Vec<u8> = buf[..info.buffer_size()].chunks_exact(n).flat_map(|c| [c[0], c[1], c[2]]).collect();
    Ok((rgb, info.width, info.height))
}
