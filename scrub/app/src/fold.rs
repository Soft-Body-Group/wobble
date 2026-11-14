//! EXPERIMENT (branch exp/scrub-fold, the `fold` cargo feature): wobble
//! hosted INSIDE Scrub, in process, on Scrub's GPU device.
//!
//! - A clip can BE a wobble comp ("wob clip", `Clip::wob`, a .wbl file).
//!   While the playhead is on it the hosted wobble has that comp open at
//!   the clip's source frame, and the record viewer shows the comp's
//!   output texture directly: no render, no file, no copy. Every edit in
//!   wobble is in the record viewer on the next redraw.
//! - The **wobble** panel on the dock IS wobble: its whole window (node
//!   graph, properties with any exposed knobs, its own viewer and gizmos)
//!   drawn into a texture and shown there, with the pointer and keys over
//!   the panel handed to it. Diving into a subwob is wobble's own Enter.
//!
//! What it does not do yet (the wrinkles left alone on purpose): playback
//! renders the wob live every frame (no baked tier), export and stills
//! show the clip's footage, a dissolve shows the wob only on its outgoing
//! side, one comp is open at a time (cutting between two wob clips swaps
//! comps at the cut), and the comp shown is whatever wobble's viewer is
//! pointed at.

use crate::editor::Editor;
use crate::timelineui::Selection;
use crate::viewer::{Idt, TexPaint};
use egui::{Event, Rect, Sense};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wobble_app::hosted::{Hosted, Seed};

const SOURCE_NODE: &str = "SCRUB SOURCE";
/// The node Scrub shows (wobble's `hosted::OUT_NODE`).
const OUT_NODE: &str = wobble_app::hosted::OUT_NODE;
/// Frames wobble's playback is started ahead by: what its own start
/// (seek, decode, upload) costs, so the two clocks then agree.
const LEAD: u64 = 2;

#[derive(Default)]
pub struct Fold {
    /// The shared device, until wobble is started on it.
    pub seed: Option<Seed>,
    pub host: Option<Hosted>,
    /// Start wobble at the next chance (between frames).
    want: bool,
    /// The setup wobble has open.
    loaded: Option<PathBuf>,
    /// Comps to make: (setup file, source path or pattern, input space).
    make: Vec<(PathBuf, String, Value)>,
    /// A wob clip is under the playhead this frame.
    live: bool,
    /// When the panel was last drawn (it may be in a torn-off window,
    /// drawn between the main window's frames).
    pane: Option<std::time::Instant>,
    /// A press that began on the panel holds the pointer until release.
    captured: bool,
    inside: bool,
    last_seek: Option<u64>,
    /// The comp frame wobble showed last frame (to see it move by itself).
    seen: Option<u64>,
    /// Scrub's playhead was just moved to follow wobble's: take the frame
    /// it lands on as sent, not as a seek to make.
    adopt: bool,
    /// Wobble is playing the comp itself, alongside Scrub's playback.
    playing: bool,
    /// When wobble's playback was last put back in step.
    resync: Option<std::time::Instant>,
    pub failed: Option<String>,
    /// Wobble has the main window (see `Editor::fold_layout`).
    pub dived: bool,
    /// Wobble is on its way out to a window of its own (the dock makes the
    /// window between frames): frames left to wait for it.
    pub popping: u32,
    /// Asked from the wobble panel's own bar, which is drawn while the
    /// dock is out on loan: taken up at the start of the next frame.
    /// 1 back to Scrub, 2 a window of its own.
    pub ask: u8,
}

/// What a comp's source node reads: a movie, or an image sequence's
/// pattern (`dir/name.####.exr`).
fn source_path(media: &Path) -> String {
    match (media::sequence::is_image(media), media::sequence::Pattern::of(media)) {
        (true, Some((p, _))) => p.dir.join(format!("{}{}{}", p.prefix, "#".repeat(p.pad.max(1)), p.suffix)).to_string_lossy().into_owned(),
        _ => media.to_string_lossy().into_owned(),
    }
}

/// A fresh comp in the hosted wobble: the clip's source alone, placed so
/// comp frame k is source frame k, read in the colour Scrub shows it in.
fn build(h: &mut Hosted, path: &Path, src: &str, input: &Value) -> Result<(), String> {
    let g = h.call("graph_get", json!({}))?;
    for id in g["nodes"].as_array().into_iter().flatten().filter_map(|n| n["id"].as_u64()) {
        h.call("node_delete", json!({ "id": id }))?;
    }
    let l = h.call("node_create", json!({ "type": "Load", "path": src, "name": SOURCE_NODE, "x": 0.0, "y": 0.0 }))?["id"].as_u64().ok_or("no Load id")?;
    h.call("node_patch", json!({ "id": l, "kind": { "Read": { "path": src, "start": 0, "input": input } } }))?;
    // The output node: what the clip shows in Scrub, whatever wobble's
    // viewer is on. Build between the two.
    let o = h.call("node_create", json!({ "type": "Dot", "inputs": [l], "name": OUT_NODE, "x": 0.0, "y": 260.0 }))?["id"].as_u64().ok_or("no output id")?;
    let _ = h.call("view_set", json!({ "id": o }));
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    h.call("setup_save", json!({ "path": path }))?;
    Ok(())
}

impl Editor {
    /// Between frames, where a window can be made: start wobble when asked.
    pub fn fold_between(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        let f = &mut self.fold;
        if f.want && f.host.is_none() {
            f.want = false;
            if let Some(seed) = f.seed.take() {
                let t0 = std::time::Instant::now();
                match Hosted::new(el, seed) {
                    Ok(h) => {
                        log::info!("hosted wobble up in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
                        f.host = Some(h);
                    }
                    Err(e) => f.failed = Some(format!("{e:#}")),
                }
            }
        }
        if let Some(h) = &mut f.host {
            h.between(el);
        }
    }

    pub fn fold_window_event(&mut self, el: &winit::event_loop::ActiveEventLoop, id: winit::window::WindowId, event: winit::event::WindowEvent) {
        if let Some(h) = &mut self.fold.host {
            h.window_event(el, id, event);
        }
    }

    /// Start of Scrub's frame: the comps asked for are made, and the comp
    /// of the wob clip under the playhead is opened at its frame.
    pub(crate) fn fold_begin(&mut self) {
        self.fold.live = false;
        let t = self.rec.peek.unwrap_or(self.rec.transport.frame() as i64);
        let at = self.rec.seq.video_at(t).and_then(|(c, f)| c.wob.clone().map(|w| (PathBuf::from(w), f.max(0) as u64, c.speed)));
        let span = self.rec.seq.video_at(t).map(|(c, _)| (c.start, c.len, c.speed));
        let mut follow: Option<i64> = None;
        // Plain forward playback: wobble plays the comp itself (its player
        // reads ahead; a seek every frame cannot keep up) and is only put
        // back when it drifts.
        let tr = &self.rec.transport;
        let run = tr.playing() && tr.dir() > 0 && self.rec.peek.is_none() && at.as_ref().is_some_and(|a| a.2 == 1.0);
        let fps = self.rec.seq.fps.as_f64();
        let at = at.map(|(p, f, _)| (p, f));
        if self.fold.host.is_none() {
            if at.is_some() || !self.fold.make.is_empty() {
                self.fold.want = true;
            }
            return;
        }
        let mut said = Vec::new();
        let f = &mut self.fold;
        let h = f.host.as_mut().expect("checked");
        for (path, src, input) in std::mem::take(&mut f.make) {
            if let Some(p) = &f.loaded {
                let _ = h.call("setup_save", json!({ "path": p }));
            }
            match build(h, &path, &src, &input) {
                Ok(()) => {
                    f.loaded = Some(path);
                    f.last_seek = None;
                    said.push("wob made: build between SCRUB SOURCE and SCRUB OUT; cmd shift W comes back to Scrub".to_string());
                }
                Err(e) => said.push(format!("wob: {e}")),
            }
        }
        if let Some((path, frame)) = at {
            if f.loaded.as_deref() != Some(path.as_path()) {
                if let Some(p) = &f.loaded {
                    let _ = h.call("setup_save", json!({ "path": p }));
                }
                match h.call("setup_load", json!({ "path": path })) {
                    Ok(_) => {
                        f.loaded = Some(path.clone());
                        f.last_seek = None;
                    }
                    Err(e) => said.push(format!("wob: cannot open {}: {e}", path.display())),
                }
            }
            if f.loaded.as_deref() == Some(path.as_path()) {
                if run {
                    let drift = (h.tick() as i64 - frame as i64).abs();
                    let frame = if f.playing { frame } else { frame + LEAD };
                    // (A seek throws its read-ahead away: not more than one
                    // a second, or it never gets going again.)
                    if std::env::var_os("SCRUB_FOLD_TRACE").is_some() && f.playing && drift > 4 && f.resync.is_none_or(|t| t.elapsed().as_millis() > 1000) {
                        println!("fold:   resync to {frame} (wobble at {}) {}", h.tick(), h.call("app_info", json!({})).map(|v| format!("playing {} playhead {}", v["playing"], v["playhead"])).unwrap_or_default());
                    }
                    if !f.playing || (drift > 4 && f.resync.is_none_or(|t| t.elapsed().as_millis() > 1000)) {
                        let _ = h.call("time_set", json!({ "frame": frame }));
                        f.resync = Some(std::time::Instant::now());
                    }
                    if !f.playing {
                        let _ = h.call("comp_set", json!({ "fps": fps, "playing": true }));
                        f.playing = true;
                    }
                    f.last_seek = None;
                } else {
                    if f.playing {
                        let _ = h.call("comp_set", json!({ "playing": false }));
                        f.playing = false;
                    }
                    // Two ways. Wobble moved its own playhead (its timeline
                    // under the pointer, or its own playback): Scrub's
                    // follows. Otherwise Scrub's moved: wobble is sent there.
                    let cur = h.tick();
                    if std::mem::take(&mut f.adopt) {
                        f.last_seek = Some(frame);
                    }
                    let theirs = f.inside || f.captured || h.playing();
                    if theirs && f.last_seek.is_some() && f.seen.is_some_and(|s| s != cur) && f.last_seek != Some(cur) {
                        follow = Some(cur as i64 - frame as i64);
                        f.adopt = true;
                    } else if f.last_seek != Some(frame) {
                        let _ = h.call("time_set", json!({ "frame": frame }));
                        f.last_seek = Some(frame);
                    }
                    f.seen = Some(cur);
                }
                f.live = true;
            }
        } else if f.playing {
            let _ = h.call("comp_set", json!({ "playing": false }));
            f.playing = false;
        }
        if let (Some(d), Some((start, len, speed))) = (follow, span) {
            let step = if speed.abs() > 1e-6 { (d as f64 / speed).round() as i64 } else { d };
            let to = (t + step).clamp(start, start + len.max(1) - 1);
            if to != t {
                self.rec.transport.seek(to.max(0) as u64);
            }
        }
        for s in said {
            self.say(s);
        }
    }

    /// End of Scrub's frame: one wobble frame, when anything shows it.
    pub(crate) fn fold_end(&mut self, ctx: &egui::Context) {
        let f = &mut self.fold;
        if let Some(h) = &mut f.host {
            if f.live || f.pane.is_some_and(|t| t.elapsed().as_millis() < 250) {
                h.frame();
                ctx.request_repaint();
            }
        } else if f.want {
            ctx.request_repaint();
        }
    }

    /// The record viewer's picture when a wob clip is under the playhead:
    /// the comp's output, its size, and its space into Scrub's working one.
    pub(crate) fn fold_ext(&self) -> Option<(wgpu::TextureView, (u32, u32), Arc<Idt>)> {
        if !self.fold.live {
            return None;
        }
        let h = self.fold.host.as_ref()?;
        let (view, size) = h.output()?;
        let working: colour::color::Colorspace = serde_json::from_value(h.working()).ok()?;
        Some((view, size, Arc::new(Idt::new(working, self.rec.seq.colour.working))))
    }

    pub(crate) fn fold_selected_is_wob(&self) -> bool {
        let Selection::Clip { track, start } = self.rec.sel else { return false };
        self.rec.seq.tracks[track].clips.iter().any(|c| c.start == start && c.wob.is_some())
    }

    /// The wob controls panel: the comp's own panel (its control nodes,
    /// arranged in wobble as a subwob's are), drawn here by wobble's panel
    /// widgets. Wobble's interface is not needed to use them.
    pub(crate) fn fold_controls(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let f = &mut self.fold;
        let words = |ui: &mut egui::Ui, s: &str| {
            ui.scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| ui.label(egui::RichText::new(s).weak().small()));
        };
        let (Some(h), true) = (f.host.as_mut(), f.live) else {
            words(ui, "park the playhead on a wob clip to see its controls");
            return;
        };
        // The comp's own panel on top (as tall as it is, up to half the
        // room), the whole node panels it publishes under it.
        let mut any = false;
        let top = ui
            .scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                egui::ScrollArea::both().id_salt("wob-controls").max_height(rect.height() * 0.5).auto_shrink([false, true]).show(ui, |ui| {
                    any = h.controls_ui(ui);
                });
            })
            .response
            .rect;
        let rest = Rect::from_min_max(egui::pos2(rect.min.x, if any { top.max.y + 6.0 } else { rect.min.y }), rect.max);
        let mut panels = false;
        if rest.height() > 40.0 {
            ui.scope_builder(egui::UiBuilder::new().max_rect(rest), |ui| {
                ui.set_clip_rect(rest.intersect(ui.clip_rect()));
                panels = h.node_panels_ui(ui);
            });
        }
        if !any && !panels {
            words(ui, "this wob publishes no controls: in wobble, open `panel` in the graph's bar, then drag property rows onto it, or select a node and press `+ selected node` for its whole panel");
        }
    }

    /// The pointer is over the wobble panel (or held by it): its keys, not
    /// Scrub's.
    pub(crate) fn fold_has_keys(&self, ui: &egui::Ui) -> bool {
        self.fold.captured || ui.input(|i| i.pointer.hover_pos()).and_then(|q| self.panes.at(q)).is_some_and(|(p, _)| p == crate::panes::Pane::Wobble)
    }

    /// The wobble panel's body: wobble's window, drawn by wobble.
    pub(crate) fn fold_pane(&mut self, ui: &mut egui::Ui, rect: Rect) {
        self.fold.pane = Some(std::time::Instant::now());
        let ppp = ui.ctx().pixels_per_point();
        let resp = ui.interact(rect, ui.id().with("wobble-guest"), Sense::click_and_drag());
        let f = &mut self.fold;
        let Some(h) = f.host.as_mut() else {
            f.want = f.seed.is_some();
            let msg = match &f.failed {
                Some(e) => format!("wobble did not start: {e}"),
                None => "starting wobble…".to_string(),
            };
            ui.painter_at(rect).text(rect.center(), egui::Align2::CENTER_CENTER, msg, egui::FontId::proportional(13.0), ui.visuals().text_color());
            return;
        };
        let over = resp.contains_pointer();
        let off = rect.min.to_vec2();
        let (events, mods) = ui.input(|i| (i.events.clone(), i.modifiers));
        let mut out = Vec::new();
        for e in events {
            match e {
                Event::PointerMoved(p) => {
                    if over || f.captured {
                        out.push(Event::PointerMoved(p - off));
                        f.inside = true;
                    } else if f.inside {
                        out.push(Event::PointerGone);
                        f.inside = false;
                    }
                }
                Event::PointerButton { pos, button, pressed, modifiers } => {
                    if pressed && over {
                        f.captured = true;
                        out.push(Event::PointerButton { pos: pos - off, button, pressed, modifiers });
                    } else if !pressed && f.captured {
                        f.captured = false;
                        out.push(Event::PointerButton { pos: pos - off, button, pressed, modifiers });
                    }
                }
                Event::PointerGone => {
                    out.push(Event::PointerGone);
                    f.inside = false;
                }
                Event::MouseWheel { .. } | Event::Zoom(_) if over => out.push(e),
                Event::Key { .. } | Event::Text(_) | Event::Copy | Event::Cut | Event::Paste(_) | Event::Ime(_) if over || f.captured => out.push(e),
                _ => {}
            }
        }
        h.resize((rect.width() * ppp).round() as u32, (rect.height() * ppp).round() as u32, ppp);
        h.input(out, mods, over || f.captured);
        if let Some(view) = h.ui_view() {
            ui.painter_at(rect).add(egui_wgpu::Callback::new_paint_callback(rect, TexPaint { view }));
        }
        if over {
            ui.ctx().set_cursor_icon(h.cursor());
        }
    }

    /// cmd shift W: the selected clip becomes a wob clip (a comp of its
    /// source, to build on), and the wobble panel opens. On a clip that
    /// already is one: just the panel.
    pub(crate) fn fold_make(&mut self) {
        let Selection::Clip { track, start } = self.rec.sel else {
            self.say("wob: select one picture clip first");
            return;
        };
        let Some(i) = self.rec.seq.tracks[track].clips.iter().position(|c| c.start == start) else { return };
        let c = &self.rec.seq.tracks[track].clips[i];
        let (title, has, media) = (c.text.is_some() || self.rec.seq.media[c.media].generated, c.wob.is_some(), self.rec.seq.media[c.media].path.clone());
        if title {
            self.say("wob: a footage clip, not a title");
            return;
        }
        self.fold_open_controls();
        self.fold_open_pane();
        if has {
            return;
        }
        let input = serde_json::to_value(colour::color::InputSpace::Explicit(self.rec.seq.colour.input_of(&media))).unwrap_or(Value::Null);
        let dir = match &self.project.path {
            Some(p) => p.with_extension("wobs"),
            None => std::env::temp_dir().join("scrub-wobs"),
        };
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
        let path = dir.join(format!("wob-{stamp:x}.wbl"));
        self.rec.seq.tracks[track].clips[i].wob = Some(path.to_string_lossy().into_owned());
        self.changed(false);
        self.fold.make.push((path, source_path(&media), input));
        self.fold.want = self.fold.host.is_none();
    }

    /// The selected clip plays its footage again (the comp file stays).
    pub(crate) fn fold_clear(&mut self) {
        let Selection::Clip { track, start } = self.rec.sel else { return };
        if let Some(c) = self.rec.seq.tracks[track].clips.iter_mut().find(|c| c.start == start) {
            if c.wob.take().is_some() {
                self.changed(false);
                self.say("wob removed: the clip plays its footage");
            }
        }
    }

    pub(crate) fn fold_save_quiet(&mut self) {
        let f = &mut self.fold;
        if let (Some(h), Some(p)) = (f.host.as_mut(), f.loaded.clone()) {
            let _ = h.call("setup_save", json!({ "path": p }));
        }
    }

    /// Save the comp wobble has open (also done whenever it swaps comps).
    pub(crate) fn fold_save(&mut self) {
        let f = &mut self.fold;
        if let (Some(h), Some(p)) = (f.host.as_mut(), f.loaded.clone()) {
            let r = h.call("setup_save", json!({ "path": p }));
            self.say(match r {
                Ok(_) => "wob saved".to_string(),
                Err(e) => format!("wob: not saved: {e}"),
            });
        }
    }
}

/// `SCRUB_TEST_FOLD=<folder>` with `--shot`: the experiment end to end,
/// headless. A two-clip cut; the first clip made a wob; the hosted wobble
/// started; a shot of the window (`a.png`: record viewer = the comp = the
/// source); a Mult x0.25 added in wobble through its API; a second shot
/// (`b.png`: the record viewer is darker, with no render and no save).
impl crate::App {
    pub(crate) fn test_fold(&mut self, el: &winit::event_loop::ActiveEventLoop, dir: &Path) {
        use std::time::{Duration, Instant};
        let (Some(gpu), Some(win)) = (&self.gpu, &mut self.win) else { return };
        let ed = &mut self.editor;
        let _ = std::fs::create_dir_all(dir);
        ed.build_cuts(1, false);
        // One long clip (the source has 240 frames), to play through.
        ed.rec.seq.tracks[model::V1].clips[0].len = 150;
        ed.rec.fit_transport();
        ed.focus_record();
        ed.rec.sel = Selection::Clip { track: model::V1, start: 0 };
        ed.rec.transport.seek(3);
        let t0 = Instant::now();
        ed.fold_make();
        let pump = |ed: &mut Editor, win: &mut imfram::shell::UiWindow, until: &dyn Fn(&Editor) -> bool, secs: u64| -> (bool, Option<wgpu::Texture>) {
            let t = Instant::now();
            let mut tex = None;
            while t.elapsed() < Duration::from_secs(secs) {
                ed.fold_between(el);
                tex = Some(win.offscreen(gpu, ed));
                if until(ed) {
                    return (true, tex);
                }
                std::thread::sleep(Duration::from_millis(15));
            }
            (false, tex)
        };
        let (up, _) = pump(ed, win, &|e| e.fold.host.is_some() && e.fold.loaded.is_some(), 60);
        println!("fold: wobble hosted and the wob made: {up} ({:.0} ms) {:?} {:?}", t0.elapsed().as_secs_f64() * 1e3, ed.fold.failed, ed.fold.loaded);
        // Let the comp's frame decode and land.
        let n0 = Instant::now();
        let (_, tex) = pump(ed, win, &|_| n0.elapsed() > Duration::from_secs(3), 5);
        if let Some(t) = &tex {
            let _ = imfram::shell::capture::save_png(gpu, t, &dir.join("a.png"));
        }
        // The pointer and keys over wobble are wobble's: click its "panel"
        // button (c.png: the panel composer opens, at the comp's top level).
        {
            use egui::{Event as E, Modifiers as M, PointerButton as B};
            let at = egui::pos2(47.0, 78.0);
            let mut tex = None;
            for evs in [vec![E::PointerMoved(at)], vec![E::PointerButton { pos: at, button: B::Primary, pressed: true, modifiers: M::NONE }], vec![E::PointerButton { pos: at, button: B::Primary, pressed: false, modifiers: M::NONE }], vec![], vec![]] {
                win.inject(evs);
                ed.fold_between(el);
                tex = Some(win.offscreen(gpu, ed));
            }
            if let Some(t) = &tex {
                let _ = imfram::shell::capture::save_png(gpu, t, &dir.join("c.png"));
            }
        }
        // In wobble: a Mult before the output node, a slider published as
        // the comp's control and cabled to it, and the VIEWER left on the
        // source. Scrub must show the output node, and the slider.
        if let Some(h) = ed.fold.host.as_mut() {
            let r = (|| -> Result<String, String> {
                let g = h.call("graph_get", json!({}))?;
                let id = |name: &str| g["nodes"].as_array().into_iter().flatten().find(|n| n["name"] == name).and_then(|n| n["id"].as_u64());
                let (l, o) = (id(SOURCE_NODE).ok_or("no source node")?, id(OUT_NODE).ok_or("no output node")?);
                let m = h.call("node_create", json!({ "type": "Mult", "inputs": [l] }))?["id"].as_u64().ok_or("no mult")?;
                h.call("node_set", json!({ "id": m, "values": { "MultR": 0.25, "MultG": 0.25, "MultB": 1.0 } }))?;
                h.call("node_connect", json!({ "src": m, "dst": o, "input": 0 }))?;
                h.call("view_set", json!({ "id": l }))?;
                let knob = (|| -> Result<(), String> {
                    let s = h.call("node_create", json!({ "type": "ui.slider", "name": "red", "x": 300.0, "y": 0.0 }))?["id"].as_u64().ok_or("no slider")?;
                    let w = h.call("node_create", json!({ "type": "ui.wheel", "name": "lift", "x": 300.0, "y": 80.0 }));
                    // As a Grade's lift pulled onto the composer makes it: an offset about 0.
                    // Three more wheels, as a Grade's four pulled onto a two-tile panel.
                    for (k, (name, v, sc)) in [("gain", 1.0, 1.0), ("gamma", 1.0, 1.0), ("offset", 0.0, 0.0)].into_iter().enumerate() {
                        let _ = h.call("node_create", json!({ "type": "ui.wheel", "name": name, "x": 300.0, "y": 160.0 + 80.0 * k as f64 }));
                        let _ = h.call("node_patch", json!({ "id": 10 + k, "kind": { "Custom": { "type_id": "ui.wheel", "params": { "r": v, "g": v, "b": v, "lo": -4.0, "hi": 8.0, "scale": sc } } } }));
                    }
                    let _ = h.call("node_patch", json!({ "id": 9, "kind": { "Custom": { "type_id": "ui.wheel", "params": { "r": 0.0, "g": 0.0, "b": 0.0, "lo": -4.0, "hi": 8.0, "scale": 0.0 } } } }));
                    println!("fold: wheel {:?}", w.map(|v| v["id"].clone()));
                    h.call("patch_link", json!({ "from": { "node": s, "channel": "Custom:value" }, "to": { "node": m, "channel": "MultR" } }))?;
                    Ok(())
                })();
                // A Transform after the Mult, its scale published as a pair.
                let pair = (|| -> Result<(), String> {
                    let t = h.call("node_create", json!({ "type": "Transform", "inputs": [m] }))?["id"].as_u64().ok_or("no transform")?;
                    h.call("node_connect", json!({ "src": t, "dst": o, "input": 0 }))?;
                    let p = h.call("node_create", json!({ "type": "ui.pair", "name": "scale", "x": 300.0, "y": 400.0 }))?["id"].as_u64().ok_or("no pair")?;
                    h.call("node_patch", json!({ "id": p, "kind": { "Custom": { "type_id": "ui.pair", "params": { "x": 1.0, "y": 1.0, "lo": 0.0, "hi": 4.0, "lock": 1.0 } } } }))?;
                    h.call("patch_link", json!({ "from": { "node": p, "channel": "Custom:x" }, "to": { "node": t, "channel": "ScaleX" } }))?;
                    h.call("patch_link", json!({ "from": { "node": p, "channel": "Custom:y" }, "to": { "node": t, "channel": "ScaleY" } }))?;
                    Ok(())
                })();
                println!("fold: scale pair {pair:?}");
                // The 2x2 arrangement the composer would make: through the file.
                if let Some(path) = h.call("app_info", json!({})).ok().and_then(|v| v["setup"].as_str().map(PathBuf::from)) {
                    h.call("setup_save", json!({ "path": path }))?;
                    let mut d: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
                    d["panel"] = json!([2, 5]);
                    // The Mult's whole panel published too.
                    d["graph"]["host_panels"] = json!([m]);
                    for (id, cell) in [("14", [0, 0]), ("8", [3, 0]), ("9", [0, 1]), ("10", [3, 1]), ("11", [0, 8]), ("12", [3, 8])] {
                        d["graph"]["nodes"][id]["ui_cell"] = json!(cell);
                    }
                    std::fs::write(&path, d.to_string()).map_err(|e| e.to_string())?;
                    h.call("setup_load", json!({ "path": path }))?;
                }
                Ok(format!("slider: {knob:?}"))
            })();
            println!("fold: a Mult before the output node, the viewer on the source: {r:?}; comp frame {} working {}", h.tick(), h.working());
        }
        // Back to Scrub (b.png): the record viewer shows the OUTPUT node
        // (darker), the wob controls panel shows the slider, no wobble UI.
        ed.fold_close_pane();
        let n1 = Instant::now();
        let (_, tex) = pump(ed, win, &|_| n1.elapsed() > Duration::from_secs(2), 4);
        if let Some(h) = ed.fold.host.as_ref() {
            println!("fold: output node present {}, its size {:?}", h.has_out(), h.output().map(|o| o.1));
        }
        if let Some(h) = ed.fold.host.as_mut() {
            let g = h.call("graph_get", json!({})).unwrap_or_default();
            println!("fold: nodes {}", g["nodes"].as_array().into_iter().flatten().map(|n| format!("{}:{}:{}", n["id"], n["type"], n["name"])).collect::<Vec<_>>().join(" "));
        }
        if let Some(t) = &tex {
            let _ = imfram::shell::capture::save_png(gpu, t, &dir.join("b.png"));
        }
        // The slider, dragged in SCRUB's controls panel (d.png: its value
        // and the record viewer's red both up).
        {
            use egui::{Event as E, Modifiers as M, PointerButton as B};
            let (from, to) = (egui::pos2(96.0, 57.0), egui::pos2(160.0, 57.0));
            let mut tex = None;
            for evs in [vec![E::PointerMoved(from)], vec![E::PointerButton { pos: from, button: B::Primary, pressed: true, modifiers: M::NONE }], vec![E::PointerMoved(egui::pos2(128.0, 57.0))], vec![E::PointerMoved(to)], vec![E::PointerButton { pos: to, button: B::Primary, pressed: false, modifiers: M::NONE }], vec![], vec![], vec![]] {
                win.inject(evs);
                ed.fold_between(el);
                tex = Some(win.offscreen(gpu, ed));
            }
            if let Some(t) = &tex {
                let _ = imfram::shell::capture::save_png(gpu, t, &dir.join("d.png"));
            }
        }
        // The wheel, dragged the same way (e.png), its values printed as
        // the drag goes.
        {
            use egui::{Event as E, Modifiers as M, PointerButton as B};
            let wheel = |ed: &mut Editor| {
                let Some(h) = ed.fold.host.as_mut() else { return String::new() };
                let r = |h: &mut Hosted, id: u64| h.call("node_get", json!({ "id": id })).ok().map(|v| format!("{:.3}", v["values"]["Custom:r"].as_f64().unwrap_or(-9.0))).unwrap_or_default();
                format!("lift r {} | gain r {} | gamma r {} | offset r {}", r(h, 9), r(h, 10), r(h, 11), r(h, 12))
            };
            println!("fold: wheel before {}", wheel(ed));
            let pts = [egui::pos2(90.0, 164.0), egui::pos2(96.0, 160.0), egui::pos2(104.0, 154.0), egui::pos2(112.0, 148.0)];
            let mut tex = None;
            let mut steps = vec![vec![E::PointerMoved(pts[0])], vec![E::PointerButton { pos: pts[0], button: B::Primary, pressed: true, modifiers: M::NONE }]];
            for p in &pts[1..] {
                steps.push(vec![E::PointerMoved(*p)]);
                steps.push(vec![]);
            }
            steps.push(vec![E::PointerButton { pos: pts[3], button: B::Primary, pressed: false, modifiers: M::NONE }]);
            steps.push(vec![]);
            steps.push(vec![]);
            for evs in steps {
                win.inject(evs);
                ed.fold_between(el);
                tex = Some(win.offscreen(gpu, ed));
                println!("fold:   wheel {}", wheel(ed));
            }
            if let Some(t) = &tex {
                let _ = imfram::shell::capture::save_png(gpu, t, &dir.join("e.png"));
            }
        }
        // A slider on the Mult's WHOLE panel, dragged in Scrub.
        {
            use egui::{Event as E, Modifiers as M, PointerButton as B};
            let mult = |ed: &mut Editor| ed.fold.host.as_mut().and_then(|h| h.call("node_get", json!({ "id": 7 })).ok()).map(|v| v["values"].to_string()).unwrap_or_default();
            // (Its red is cabled to the slider above: a cabled lane holds a
            // ganged triple still, so the cable comes off first.)
            if let Some(h) = ed.fold.host.as_mut() {
                let _ = h.call("patch_unlink", json!({ "to": { "node": 7, "channel": "MultR" } }));
            }
            for _ in 0..3 {
                ed.fold_between(el);
                let _ = win.offscreen(gpu, ed);
            }
            let before = mult(ed);
            let (from, to) = (egui::pos2(195.0, 639.0), egui::pos2(165.0, 639.0));
            for evs in [vec![E::PointerMoved(from)], vec![E::PointerButton { pos: from, button: B::Primary, pressed: true, modifiers: M::NONE }], vec![E::PointerMoved(egui::pos2(180.0, 639.0))], vec![E::PointerMoved(to)], vec![E::PointerButton { pos: to, button: B::Primary, pressed: false, modifiers: M::NONE }], vec![], vec![]] {
                win.inject(evs);
                ed.fold_between(el);
                let _ = win.offscreen(gpu, ed);
            }
            println!("fold: the Mult's own blue slider dragged in Scrub: {before} -> {}", mult(ed));
            let kind = |ed: &mut Editor| ed.fold.host.as_mut().and_then(|h| h.call("node_get", json!({ "id": 7 })).ok()).map(|v| v["kind"].to_string()).unwrap_or_default();
            let k0 = kind(ed);
            let at = egui::pos2(22.0, 674.0);
            for evs in [vec![E::PointerMoved(at)], vec![E::PointerButton { pos: at, button: B::Primary, pressed: true, modifiers: M::NONE }], vec![E::PointerButton { pos: at, button: B::Primary, pressed: false, modifiers: M::NONE }], vec![], vec![]] {
                win.inject(evs);
                ed.fold_between(el);
                let _ = win.offscreen(gpu, ed);
            }
            println!("fold: its 'alpha too' box clicked in Scrub: {k0} -> {}", kind(ed));
        }
        // Two-way playheads: wobble moves its own (as its timeline under
        // the pointer does) and Scrub's follows.
        {
            let t0 = ed.rec.transport.frame();
            ed.fold.inside = true;
            if let Some(h) = ed.fold.host.as_mut() {
                let at = h.tick();
                let _ = h.call("time_set", json!({ "frame": at + 10 }));
            }
            let n = Instant::now();
            let _ = pump(ed, win, &|_| n.elapsed() > Duration::from_millis(1200), 3);
            let t1 = ed.rec.transport.frame();
            ed.fold.inside = false;
            ed.rec.transport.seek(t1 + 5);
            let n = Instant::now();
            let _ = pump(ed, win, &|_| n.elapsed() > Duration::from_millis(1200), 3);
            let want = ed.rec.seq.video_at(ed.rec.transport.frame() as i64).map(|(_, f)| f);
            println!("fold: wobble moved 10 frames on: Scrub's playhead {t0} -> {t1}; Scrub moved 5 on: wobble at {:?}, Scrub wants {want:?}", ed.fold.host.as_ref().map(|h| h.tick()));
        }
        // Playback through the wob clip, paced at 24 a second: how far the
        // comp frame wobble shows trails the frame Scrub asks for.
        ed.rec.transport.seek(0);
        for _ in 0..10 {
            ed.fold_between(el);
            let _ = win.offscreen(gpu, ed);
            std::thread::sleep(Duration::from_millis(30));
        }
        ed.rec.transport.toggle();
        if let Some(h) = ed.fold.host.as_mut() {
            println!("fold: app_info {}", h.call("app_info", json!({})).map(|v| v.to_string()).unwrap_or_default().chars().take(600).collect::<String>());
        }
        let (mut n, mut exact, mut late) = (0u32, 0u32, 0i64);
        let (mut n2, mut exact2, mut near2, mut late2) = (0u32, 0u32, 0u32, 0i64);
        let tp = Instant::now();
        while ed.rec.transport.playing() && ed.rec.transport.frame() < 140 && tp.elapsed() < Duration::from_secs(8) {
            let t = Instant::now();
            ed.fold_between(el);
            let _ = win.offscreen(gpu, ed);
            let want = ed.rec.seq.video_at(ed.rec.transport.frame() as i64).map(|(_, f)| f);
            if let (Some(w), Some(h)) = (want, ed.fold.host.as_ref()) {
                n += 1;
                let d = w - h.tick() as i64;
                if n % 12 == 0 || std::env::var_os("SCRUB_FOLD_TRACE").is_some() {
                    println!("fold:   scrub wants {w}, wobble shows {}", h.tick());
                }
                exact += (d == 0) as u32;
                late = late.max(d.abs());
                if tp.elapsed() > Duration::from_secs(1) {
                    n2 += 1;
                    exact2 += (d == 0) as u32;
                    near2 += (d.abs() <= 1) as u32;
                    late2 = late2.max(d.abs());
                }
            }
            std::thread::sleep(Duration::from_secs_f64(1.0 / 60.0).saturating_sub(t.elapsed()));
        }
        ed.rec.transport.stop();
        println!("fold: playing the wob clip live (24 fps, 60 redraws a second): {n} redraws, {exact} showing the exact comp frame, worst {late} frames off");
        println!("fold: after the first second: {n2} redraws, {exact2} exact, {near2} within a frame, worst {late2} off");
        // Dive again, then out to a window of its own: it must stay there.
        ed.fold_make();
        for _ in 0..5 {
            ed.dock_between(el, gpu);
            ed.fold_between(el);
            let _ = win.offscreen(gpu, ed);
        }
        let dived = ed.fold_where();
        ed.fold_own_window();
        for _ in 0..40 {
            ed.dock_between(el, gpu);
            ed.fold_between(el);
            let _ = win.offscreen(gpu, ed);
        }
        let name = |w: u8| ["closed", "the whole main window", "a window of its own"][w as usize];
        println!("fold: dived: wobble is {}; after own window: {}", name(dived), name(ed.fold_where()));
        println!("fold: shots in {}", dir.display());
    }
}
