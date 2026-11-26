//! The scripting protocol's app side (`api` holds the table, the socket and
//! the MCP bridge). Calls are drained at the top of each UI pass, so a call
//! runs between frames on the UI thread, exactly where a key would, and
//! every edit goes through the same model operations the keys and the mouse
//! use.
//!
//! Undo: while a call runs, the edit helpers leave the history alone
//! (`api_quiet`); afterwards a call that changed the sequence becomes ONE
//! undo step, and a failed call (or a failed `batch`, at any of its calls)
//! puts the sequence back as it was. A key burst still open when a call
//! arrives (a trim key held) is closed first, so it stays its own step.
//!
//! `viewer_capture` composes its frame on a worker (`render::still`, the
//! renderer's own composition) and answers from there.

use crate::editor::{Editor, Focus};
use crate::timelineui::{Selection, Tool};
use api::server::Call;
use api::ApiError;
use model::{Side, TRACKS};
use serde_json::{json, Value};
use std::path::PathBuf;

type R = Result<Value, ApiError>;

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError::invalid(msg)
}

fn no(msg: impl Into<String>) -> ApiError {
    ApiError::app(msg)
}

fn int(p: &Value, k: &str) -> Option<i64> {
    p.get(k).and_then(Value::as_i64)
}

fn num(p: &Value, k: &str) -> Option<f64> {
    p.get(k).and_then(Value::as_f64)
}

fn boolean(p: &Value, k: &str) -> Option<bool> {
    p.get(k).and_then(Value::as_bool)
}

fn string<'a>(p: &'a Value, k: &str) -> Option<&'a str> {
    p.get(k).and_then(Value::as_str)
}

fn need_int(p: &Value, k: &str) -> Result<i64, ApiError> {
    int(p, k).ok_or_else(|| bad(format!("'{k}' must be an integer")))
}

/// A track by its name ("V1" … "M").
fn track_named(name: &str) -> Result<usize, ApiError> {
    TRACKS.iter().position(|t| t.0.eq_ignore_ascii_case(name)).ok_or_else(|| bad(format!("no track {name} (V1 V2 V3 A1 A2 A3 M)")))
}

fn track_of(p: &Value, k: &str) -> Result<usize, ApiError> {
    track_named(string(p, k).ok_or_else(|| bad(format!("'{k}' must be a track name")))?)
}

fn side_of(p: &Value) -> Result<Side, ApiError> {
    match string(p, "side").unwrap_or("roll") {
        "out" => Ok(Side::Out),
        "in" => Ok(Side::In),
        "roll" => Ok(Side::Roll),
        s => Err(bad(format!("side {s}: out, in or roll"))),
    }
}

fn side_name(s: Side) -> &'static str {
    match s {
        Side::Out => "out",
        Side::In => "in",
        Side::Roll => "roll",
    }
}

fn tool_name(t: Tool) -> &'static str {
    match t {
        Tool::Select => "select",
        Tool::Razor => "razor",
        Tool::Rate => "rate",
        Tool::Slip => "slip",
        Tool::Slide => "slide",
        Tool::Roll => "roll",
    }
}

/// Methods a `batch` refuses.
const NOT_IN_BATCH: &[&str] = &["undo", "redo", "batch", "project_open", "project_new", "post_open", "viewer_capture"];

impl Editor {
    /// Start listening (127.0.0.1:`port`, 0 picks one). The port, or why not.
    pub fn api_listen(&mut self, port: u16) -> Result<u16, String> {
        let h = api::server::serve(port).map_err(|e| format!("the scripting API could not listen on 127.0.0.1:{port}: {e}"))?;
        let p = h.port;
        self.api = Some(h);
        Ok(p)
    }

    /// Answer every call that arrived since the last UI pass.
    pub(crate) fn api_drain(&mut self) {
        let Some(host) = self.api.take() else { return };
        let calls: Vec<Call> = host.rx.try_iter().collect();
        self.api = Some(host);
        for call in calls {
            if call.method == "viewer_capture" {
                self.api_capture(call);
                continue;
            }
            let r = self.api_recorded(&call.method, &call.params);
            call.answer(r);
        }
    }

    /// One call as one undo step: the history is left alone while it runs,
    /// a change is committed once after, a failure is undone.
    pub(crate) fn api_recorded(&mut self, method: &str, params: &Value) -> R {
        let mutates = api::method(method).is_some_and(|m| m.mutates) && method != "undo" && method != "redo";
        if !mutates {
            return self.api_call(method, params);
        }
        if self.rec.history.pending() {
            self.rec.history.commit(&self.rec.seq);
        }
        let before = self.rec.seq.clone();
        self.api_quiet = true;
        let r = self.api_call(method, params);
        self.api_quiet = false;
        match &r {
            Ok(_) if self.rec.seq != before => {
                self.rec.history.touched();
                self.rec.history.commit(&self.rec.seq);
            }
            Ok(_) => {}
            Err(_) => self.rec.seq = before,
        }
        self.rec.fit_transport();
        r
    }

    fn clip_index(&self, track: usize, start: i64) -> Result<usize, ApiError> {
        self.rec.seq.tracks[track]
            .clips
            .iter()
            .position(|c| c.start == start)
            .ok_or_else(|| no(format!("no clip on {} starting at {start} (timeline_get lists them)", self.rec.seq.tracks[track].name)))
    }

    fn clip_param(&self, p: &Value) -> Result<(usize, usize), ApiError> {
        let t = track_of(p, "track")?;
        Ok((t, self.clip_index(t, need_int(p, "start")?)?))
    }

    /// Every method in the table but `viewer_capture`.
    pub(crate) fn api_call(&mut self, method: &str, p: &Value) -> R {
        match method {
            "api_schema" => Ok(api::schema()),
            "app_info" => Ok(self.api_info()),
            "timeline_get" => Ok(self.api_timeline()),
            "bin_list" => Ok(json!({ "items": self.bin.items.iter().map(|i| json!({ "path": i.path, "name": i.name, "detail": i.detail, "group": i.group, "range": i.range.map(|(a, b)| json!({ "in": a, "out": b })), "rating": rating_name(i.rating) })).collect::<Vec<_>>() })),
            "mode_set" => {
                let want = match string(p, "mode") {
                    Some("edit") => crate::editor::Mode::Edit,
                    Some("selects") => crate::editor::Mode::Selects,
                    _ => return Err(bad("mode: edit or selects")),
                };
                if self.mode != want {
                    self.toggle_mode();
                }
                Ok(json!({ "mode": mode_name(self.mode) }))
            }
            "selects_list" => {
                let f = filter_of(p)?;
                let clips: Vec<Value> = self
                    .bin
                    .listed(f)
                    .into_iter()
                    .map(|i| {
                        let it = &self.bin.items[i];
                        let m = self.remembered_marks(it);
                        let subs = self.bin.items.iter().filter(|x| x.path == it.path && x.range.is_some()).count();
                        json!({ "path": it.path, "name": it.name, "group": it.group, "rating": rating_name(it.rating), "marks": { "in": m.inp, "out": m.out }, "subclips": subs })
                    })
                    .collect();
                let counts: serde_json::Map<String, Value> = crate::bin::Filter::ALL.iter().map(|f| (f.name().to_string(), json!(self.bin.listed(*f).len()))).collect();
                Ok(json!({ "filter": f.name(), "clips": clips, "counts": counts }))
            }
            "clip_rate" => {
                let r = match string(p, "rating") {
                    Some("select") => Some(crate::bin::Rating::Select),
                    Some("reject") => Some(crate::bin::Rating::Reject),
                    Some("none") => None,
                    _ => return Err(bad("rating: select, reject or none")),
                };
                let path = match string(p, "path") {
                    Some(_) => path_of(p, "path"),
                    None => self.source.as_ref().map(|c| c.original.clone()).ok_or_else(|| no("no source clip: name one with 'path'"))?,
                };
                let it = self.bin.clip_mut(&path).ok_or_else(|| no(format!("{} is not in the bin (bin_add it first)", path.display())))?;
                it.rating = r;
                Ok(json!({ "path": path, "rating": rating_name(r) }))
            }
            "subclip_add" => {
                if string(p, "path").is_some() {
                    let path = path_of(p, "path");
                    if self.source.as_ref().map(|c| &c.original) != Some(&path) {
                        self.error = None;
                        self.load_item(&path, None);
                    }
                    if self.source.as_ref().map(|c| &c.original) != Some(&path) {
                        return Err(no(self.error.clone().unwrap_or_else(|| format!("could not open {}", path.display()))));
                    }
                }
                let Some(c) = self.source.as_mut() else { return Err(no("no source clip: name one with 'path'")) };
                let last = c.info.frames as i64 - 1;
                let (a, b) = (int(p, "in").or(c.marks.inp).unwrap_or(0), int(p, "out").or(c.marks.out).unwrap_or(last));
                if a < 0 || b > last || a > b {
                    return Err(bad(format!("in {a} .. out {b}: the clip's frames are 0..{last}, in before out")));
                }
                // Through the same path as M: marks set, then added.
                let keep = c.marks;
                c.marks = model::Marks { inp: Some(a), out: Some(b) };
                let n = self.bin.items.len();
                self.add_subclip();
                if let Some(c) = self.source.as_mut() {
                    c.marks = keep;
                }
                if self.bin.items.len() == n {
                    return Err(no("that in..out is in the selects bin already"));
                }
                let it = &self.bin.items[self.bin.sel];
                Ok(json!({ "path": it.path, "name": it.name, "in": a, "out": b }))
            }
            "subclip_list" => {
                let r = self.bin.selects_range();
                Ok(json!({ "subclips": self.bin.items[r].iter().map(|it| { let (a, b) = it.range.unwrap_or((0, 0)); json!({ "path": it.path, "name": it.name, "in": a, "out": b, "frames": b - a + 1 }) }).collect::<Vec<_>>() }))
            }
            "subclip_remove" => {
                let key = (path_of(p, "path"), Some((need_int(p, "in")?, need_int(p, "out")?)));
                let i = self.bin.items.iter().position(|it| it.key() == key).ok_or_else(|| no("no such subclip (subclip_list names them)"))?;
                self.bin.items.remove(i);
                self.bin.chosen.remove(&key);
                self.bin.sel = self.bin.sel.min(self.bin.items.len().saturating_sub(1));
                Ok(json!({ "removed": true, "subclips": self.bin.selects_range().len() }))
            }
            "subclips_place" => {
                let picks: Vec<(PathBuf, Option<(i64, i64)>)> = match p.get("subclips").and_then(Value::as_array) {
                    Some(v) => v
                        .iter()
                        .map(|s| Ok((path_of(s, "path"), Some((need_int(s, "in")?, need_int(s, "out")?)))))
                        .collect::<Result<_, ApiError>>()?,
                    None => self.bin.items[self.bin.selects_range()].iter().map(|it| it.key()).collect(),
                };
                if picks.is_empty() {
                    return Err(no("the selects bin is empty (subclip_add first)"));
                }
                let track = match string(p, "track") {
                    Some(t) => track_named(t)?,
                    None => model::V1,
                };
                let at = int(p, "at").unwrap_or(self.rec.seq.duration().max(0));
                let insert = boolean(p, "insert").unwrap_or(false);
                let keep = (self.focus, self.monitor, self.bin.sel, self.bin.chosen.clone());
                let n = picks.len();
                self.drop_on_timeline(crate::timelineui::Payload::Clips(picks), at, Some(track), insert);
                (self.focus, self.monitor, self.bin.sel, self.bin.chosen) = keep;
                Ok(json!({ "placed": n, "from": at, "track": self.rec.seq.tracks[track].name, "duration": self.rec.seq.duration() }))
            }
            "bin_add" => {
                let path = path_of(p, "path");
                let group = string(p, "group").unwrap_or("");
                let n0 = self.bin.items.len();
                if path.is_dir() {
                    let mut items = Vec::new();
                    crate::bin::scan_tree(&path, &path, group, 2, &mut items);
                    for it in items {
                        self.bin.add_to(&it.path, it.detail, &it.group);
                    }
                } else if path.is_file() {
                    self.bin.add_to(&path, String::new(), group);
                } else {
                    return Err(no(format!("{} does not exist", path.display())));
                }
                Ok(json!({ "added": self.bin.items.len() - n0, "items": self.bin.items.len() }))
            }
            "media_open" => {
                let path = path_of(p, "path");
                self.error = None;
                self.open(&path);
                match &self.source {
                    Some(c) if c.original == path => Ok(json!({ "path": path, "frames": c.info.frames, "fps": c.info.fps(), "width": c.info.width, "height": c.info.height, "has_audio": c.info.has_audio })),
                    _ => Err(no(self.error.clone().unwrap_or_else(|| format!("could not open {}", path.display())))),
                }
            }
            "project_open" => {
                let path = path_of(p, "path");
                self.open_project(&path);
                if self.project.path.as_deref() == Some(path.as_path()) {
                    Ok(json!({ "opened": path, "offline": self.project.missing.len() }))
                } else {
                    Err(no(self.last_notice()))
                }
            }
            "project_new" => {
                self.new_project(model::Sequence::new(model::Fps::new(24, 1)), None);
                Ok(json!({ "ok": true }))
            }
            "project_save" => {
                let path = match string(p, "path") {
                    Some(_) => path_of(p, "path"),
                    None => self.project.path.clone().ok_or_else(|| no("the project was never saved: give a path"))?,
                };
                self.save_to(&path).map_err(no)?;
                Ok(json!({ "saved": self.project.path }))
            }
            "post_open" => {
                let post = path_of(p, "post");
                let rushes: Vec<PathBuf> = p.get("rushes").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).map(imfram::filebrowser::expand_home).collect();
                self.open_post_path(&post, &rushes).map_err(no)?;
                Ok(json!({ "post": self.post.as_ref().map(|q| q.title.clone()), "project": self.project.path }))
            }
            "sequence_set" => {
                if let Some(r) = p.get("raster").and_then(Value::as_array) {
                    let (w, h) = (r.first().and_then(Value::as_u64), r.get(1).and_then(Value::as_u64));
                    let (Some(w), Some(h)) = (w, h) else { return Err(bad("raster is [width, height]")) };
                    self.rec.seq.raster = Some(((w as u32).clamp(16, 16384), (h as u32).clamp(16, 16384)));
                }
                if let Some(f) = p.get("fps").and_then(Value::as_array) {
                    let (Some(n), Some(d)) = (f.first().and_then(Value::as_i64), f.get(1).and_then(Value::as_i64)) else { return Err(bad("fps is [num, den]")) };
                    if n <= 0 || d <= 0 {
                        return Err(bad("fps must be positive"));
                    }
                    if !self.rec.seq.is_empty() {
                        return Err(no("the rate is fixed once the sequence has clips"));
                    }
                    self.rec.seq.fps = model::Fps::new(n, d);
                }
                if let Some(c) = boolean(p, "conform") {
                    self.rec.seq.conform = c;
                }
                self.changed(false);
                Ok(json!({ "raster": self.rec.seq.raster, "fps": [self.rec.seq.fps.num, self.rec.seq.fps.den], "conform": self.rec.seq.conform }))
            }
            "marks_set" => {
                let source = string(p, "side") == Some("source");
                let marks = if source {
                    &mut self.source.as_mut().ok_or_else(|| no("no clip in the source viewer"))?.marks
                } else {
                    &mut self.rec.marks
                };
                if boolean(p, "clear") == Some(true) {
                    marks.clear();
                }
                if let Some(v) = p.get("in") {
                    marks.inp = v.as_i64();
                }
                if let Some(v) = p.get("out") {
                    marks.out = v.as_i64();
                }
                if let (Some(i), Some(o)) = (marks.inp, marks.out) {
                    if o < i {
                        return Err(bad(format!("out ({o}) is before in ({i})")));
                    }
                }
                Ok(json!({ "in": marks.inp, "out": marks.out }))
            }
            "playhead_set" => {
                let f = need_int(p, "frame")?.max(0) as u64;
                if string(p, "side") == Some("source") {
                    let c = self.source.as_mut().ok_or_else(|| no("no clip in the source viewer"))?;
                    c.transport.stop();
                    c.transport.seek(f);
                    Ok(json!({ "frame": c.transport.frame() }))
                } else {
                    self.rec.transport.stop();
                    self.rec.fit_transport();
                    self.rec.transport.frames = self.rec.transport.frames.max(f + 1);
                    self.rec.transport.seek(f);
                    self.focus = Focus::Record;
                    self.monitor = Focus::Record;
                    Ok(json!({ "frame": self.rec.transport.frame() }))
                }
            }
            "transport" => {
                let source = string(p, "side") == Some("source");
                let t = if source {
                    self.monitor = Focus::Source;
                    &mut self.source.as_mut().ok_or_else(|| no("no clip in the source viewer"))?.transport
                } else {
                    self.monitor = Focus::Record;
                    &mut self.rec.transport
                };
                match string(p, "action").unwrap_or_default() {
                    "play" => t.set_speed(1.0),
                    "stop" => t.stop(),
                    "toggle" => t.toggle(),
                    "shuttle_forward" => t.shuttle_forward(),
                    "shuttle_back" => t.shuttle_back(),
                    "step" => t.step_by(int(p, "frames").unwrap_or(1)),
                    "start" => {
                        t.stop();
                        t.seek(0);
                    }
                    "end" => {
                        t.stop();
                        t.seek(t.last());
                    }
                    a => return Err(bad(format!("unknown action {a}"))),
                }
                Ok(json!({ "frame": t.frame(), "speed": t.speed }))
            }
            "targets_set" => {
                let mut t = self.rec.targets;
                if let Some(v) = p.get("video") {
                    let v = match v.as_str() {
                        Some(n) => Some(track_named(n)?).filter(|t| model::VIDEO_TRACKS.contains(t)).map(Some).ok_or_else(|| bad("video targets V1-V3"))?,
                        None => None,
                    };
                    if p.get("audio").is_some() {
                        t.video = v;
                    } else {
                        t.set_video(v);
                    }
                }
                if let Some(a) = p.get("audio") {
                    t.audio = match a.as_str() {
                        Some(n) => Some(track_named(n)?).filter(|t| model::AUDIO_TRACKS.contains(t)).map(Some).ok_or_else(|| bad("audio targets A1-A3 or M"))?,
                        None => None,
                    };
                }
                self.rec.targets = t;
                Ok(self.api_targets())
            }
            "track_set" => {
                let ti = track_of(p, "track")?;
                let tr = &mut self.rec.seq.tracks[ti];
                if tr.kind == model::Kind::Video && (p.get("mute").is_some() || p.get("solo").is_some() || p.get("gain_db").is_some()) {
                    return Err(no(format!("{} is a picture track: mute, solo and gain are for A1-A3 and M", tr.name)));
                }
                if let Some(b) = boolean(p, "mute") {
                    tr.mute = b;
                }
                if let Some(b) = boolean(p, "solo") {
                    tr.solo = b;
                }
                if let Some(g) = num(p, "gain_db") {
                    tr.gain_db = (g.clamp(-60.0, 12.0) * 10.0).round() as f32 / 10.0;
                }
                let out = json!({ "track": tr.name, "mute": tr.mute, "solo": tr.solo, "gain_db": tr.gain_db });
                self.changed(false);
                Ok(out)
            }
            "edit" => {
                if self.source.is_none() {
                    return Err(no("load a clip into the source viewer first (media_open)"));
                }
                let insert = string(p, "mode") == Some("insert");
                self.error = None;
                self.notice = None;
                self.edit(insert);
                if let Some(e) = self.error.clone() {
                    return Err(no(e));
                }
                let n = self.last_notice();
                Ok(json!({ "playhead": self.rec.transport.frame(), "duration": self.rec.seq.duration(), "fitted": n.contains("fit to fill").then_some(n) }))
            }
            "fit_to_fill" => {
                let c = self.source.as_ref().ok_or_else(|| no("load a clip into the source viewer first (media_open)"))?;
                if c.marks.inp.is_none() || c.marks.out.is_none() || self.rec.marks.inp.is_none() || self.rec.marks.out.is_none() {
                    return Err(no("fit to fill needs an in and an out on both the source and the record side"));
                }
                self.fit_to_fill();
                Ok(json!({ "result": self.last_notice(), "duration": self.rec.seq.duration() }))
            }
            "select" => {
                if let Some(e) = p.get("edit") {
                    let track = track_of(e, "track")?;
                    let at = need_int(e, "at")?;
                    let side = side_of(e)?;
                    self.rec.sel = Selection::Edit { track, at, side };
                } else if let Some(cs) = p.get("clips").and_then(Value::as_array) {
                    let mut v = Vec::new();
                    for c in cs {
                        let t = track_of(c, "track")?;
                        let s = need_int(c, "start")?;
                        self.clip_index(t, s)?;
                        v.push((t, s));
                    }
                    self.rec.sel = match v.len() {
                        0 => Selection::None,
                        1 => Selection::Clip { track: v[0].0, start: v[0].1 },
                        _ => Selection::Clips(v),
                    };
                } else {
                    self.rec.sel = Selection::None;
                }
                Ok(self.api_selection())
            }
            "tool_set" => {
                let t = match string(p, "tool").unwrap_or_default() {
                    "select" => Tool::Select,
                    "razor" => Tool::Razor,
                    "rate" => Tool::Rate,
                    "slip" => Tool::Slip,
                    "slide" => Tool::Slide,
                    "roll" => Tool::Roll,
                    t => return Err(bad(format!("unknown tool {t}"))),
                };
                self.set_tool(t);
                Ok(json!({ "tool": tool_name(t) }))
            }
            "trim" => {
                let track = track_of(p, "track")?;
                let (at, d, side) = (need_int(p, "at")?, need_int(p, "frames")?, side_of(p)?);
                let ripple = boolean(p, "ripple").unwrap_or(self.rec.ripple) && side != Side::Roll;
                let moved = self.rec.seq.trim(track, at, side, d, ripple).map_err(|e| no(e.0))?;
                self.changed(false);
                let at2 = if side == Side::In && ripple { at } else { at + moved };
                Ok(json!({ "moved": moved, "cut": at2 }))
            }
            "slip" => {
                let (t, i) = self.clip_param(p)?;
                let d = self.rec.seq.slip(t, i, need_int(p, "frames")?);
                self.changed(false);
                Ok(json!({ "slipped": d, "src_in": self.rec.seq.tracks[t].clips[i].src_in }))
            }
            "slide" => {
                let (t, i) = self.clip_param(p)?;
                let d = self.rec.seq.slide(t, i, need_int(p, "frames")?);
                self.changed(false);
                let start = self.rec.seq.tracks[t].clips.iter().find(|c| c.start == need_int(p, "start").unwrap_or(0) + d).map(|c| c.start);
                Ok(json!({ "slid": d, "start": start }))
            }
            "razor" => {
                let at = need_int(p, "at")?;
                let tracks: Vec<usize> = match p.get("tracks").and_then(Value::as_array) {
                    Some(ts) if boolean(p, "all") != Some(true) && !ts.is_empty() => ts.iter().map(|t| track_named(t.as_str().unwrap_or_default())).collect::<Result<_, _>>()?,
                    _ => (0..self.rec.seq.tracks.len()).collect(),
                };
                let n = self.rec.seq.razor_tracks(&tracks, at);
                if n == 0 {
                    return Err(no(format!("nothing to cut at {at} on those tracks")));
                }
                self.changed(false);
                Ok(json!({ "cut_tracks": n }))
            }
            "clips_move" => {
                let cs = p.get("clips").and_then(Value::as_array).ok_or_else(|| bad("clips is a list of {track, start}"))?;
                let alone = boolean(p, "alone").unwrap_or(false);
                let mut picks = Vec::new();
                for c in cs {
                    let t = track_of(c, "track")?;
                    let i = self.clip_index(t, need_int(c, "start")?)?;
                    let g = if alone { vec![(t, i)] } else { self.rec.seq.group(t, i) };
                    for x in g {
                        if !picks.contains(&x) {
                            picks.push(x);
                        }
                    }
                }
                let (df, dr) = (int(p, "frames").unwrap_or(0), int(p, "rows").unwrap_or(0) as i32);
                let insert = boolean(p, "insert").unwrap_or(false);
                let (dt, dtr) = if boolean(p, "duplicate") == Some(true) {
                    self.rec.seq.duplicate_clips(&picks, df, dr, insert)
                } else {
                    self.rec.seq.move_clips(&picks, df, dr, insert)
                };
                self.rec.sel = Selection::None;
                self.changed(false);
                Ok(json!({ "frames": dt, "rows": dtr }))
            }
            "clip_delete" => {
                let (t, i) = self.clip_param(p)?;
                if boolean(p, "ripple") == Some(true) {
                    self.rec.seq.ripple_delete(t, i).map_err(|e| no(e.0))?;
                } else {
                    self.rec.seq.lift_clip(t, i);
                }
                self.rec.sel = Selection::None;
                self.changed(false);
                Ok(json!({ "duration": self.rec.seq.duration() }))
            }
            "clip_speed" => {
                let (t, i) = self.clip_param(p)?;
                let s = num(p, "speed").unwrap_or(1.0);
                if s == 0.0 || !s.is_finite() || s.abs() > crate::transport::MAX_SPEED {
                    return Err(bad("speed must be non-zero, at most 16 either way"));
                }
                let len = self.rec.seq.set_speed(t, i, s);
                self.changed(false);
                Ok(json!({ "length": len, "speed": s }))
            }
            "clip_conform" => {
                let (t, i) = self.clip_param(p)?;
                let on = boolean(p, "on").unwrap_or(true);
                let len = self.rec.seq.set_conform(t, i, on);
                self.changed(false);
                Ok(json!({ "length": len, "conform": on }))
            }
            "clip_stretch" => {
                let (t, i) = self.clip_param(p)?;
                let side = if string(p, "side") == Some("in") { Side::In } else { Side::Out };
                let (cur, s0, e0) = { let c = &self.rec.seq.tracks[t].clips[i]; (c.len, c.start, c.end()) };
                let want = need_int(p, "length")?.max(1);
                let d = if side == Side::In { cur - want } else { want - cur };
                let len = self.rec.seq.stretch(t, i, side, d);
                let start = if side == Side::In { e0 - len } else { s0 };
                let speed = self.rec.seq.tracks[t].clips.iter().find(|c| c.start == start).map(|c| c.speed);
                self.changed(false);
                Ok(json!({ "length": len, "start": start, "speed": speed, "clamped": len != want }))
            }
            "dissolve_add" => {
                let track = track_of(p, "track")?;
                let at = need_int(p, "at")?;
                let want = int(p, "length").unwrap_or(crate::edits::DISSOLVE_FRAMES).max(1);
                let d = self.rec.seq.add_dissolve(track, at, want).map_err(|e| no(e.0))?;
                self.changed(false);
                Ok(json!({ "before": d.before, "after": d.after, "length": d.len(), "clamped": d.len() < want }))
            }
            "dissolve_set" => {
                let track = track_of(p, "track")?;
                let d = self.rec.seq.set_dissolve_sides(track, need_int(p, "at")?, need_int(p, "before")?, need_int(p, "after")?).map_err(|e| no(e.0))?;
                self.changed(false);
                Ok(json!({ "before": d.before, "after": d.after, "length": d.len() }))
            }
            "dissolve_remove" => {
                let track = track_of(p, "track")?;
                let at = need_int(p, "at")?;
                if !self.rec.seq.remove_dissolve(track, at) {
                    return Err(no(format!("no dissolve on the {} cut at {at}", self.rec.seq.tracks[track].name)));
                }
                self.changed(false);
                Ok(json!({ "ok": true }))
            }
            "reframe_set" => {
                let (t, i) = self.clip_param(p)?;
                let mut r = if boolean(p, "reset") == Some(true) { model::Reframe::default() } else { self.rec.seq.tracks[t].clips[i].reframe };
                if let Some(b) = boolean(p, "fill") {
                    r.fill = b;
                }
                if let Some(z) = num(p, "zoom") {
                    r.zoom = z.clamp(0.05, 20.0);
                }
                if let Some(x) = num(p, "x") {
                    r.x = x;
                }
                if let Some(y) = num(p, "y") {
                    r.y = y;
                }
                self.rec.seq.set_reframe(t, i, r);
                self.changed(false);
                Ok(json!({ "fill": r.fill, "zoom": r.zoom, "x": r.x, "y": r.y }))
            }
            "text_create" => {
                self.rec.seq.ensure_styles();
                if let Some(n) = string(p, "style") {
                    self.textv.style = self.rec.seq.style_named(n).ok_or_else(|| no(format!("no style {n} (style_list)")))?;
                }
                let track = match p.get("track") {
                    Some(_) => Some(track_of(p, "track")?).filter(|t| model::VIDEO_TRACKS.contains(t)).map(Some).ok_or_else(|| bad("text goes on V1-V3"))?,
                    None => None,
                };
                let open = self.textv.open;
                let (t, st) = self.new_text(track, int(p, "at"), string(p, "text").unwrap_or_default()).ok_or_else(|| no("could not place the text"))?;
                self.textv.open = open;
                if let Some(len) = int(p, "length") {
                    let i = self.clip_index(t, st)?;
                    let cur = self.rec.seq.tracks[t].clips[i].len;
                    let _ = self.rec.seq.trim(t, st + cur, Side::Out, len.max(1) - cur, false);
                }
                Ok(json!({ "track": TRACKS[t].0, "start": st }))
            }
            "text_set" => {
                let (t, i) = self.clip_param(p)?;
                let style = match string(p, "style") {
                    Some(n) => Some(self.rec.seq.style_named(n).ok_or_else(|| no(format!("no style {n} (style_list)")))?),
                    None => None,
                };
                let c = self.rec.seq.tracks[t].clips[i].text.as_mut().ok_or_else(|| no("that clip is not a text clip"))?;
                if let Some(x) = string(p, "text") {
                    c.text = x.to_string();
                }
                if let Some(si) = style {
                    c.style = si;
                }
                let out = json!({ "text": c.text, "style": self.rec.seq.styles.get(c.style).map(|s| s.name.clone()) });
                self.changed(false);
                Ok(out)
            }
            "style_list" => {
                let seq = &self.rec.seq;
                let users = |k: usize| seq.tracks.iter().flat_map(|t| &t.clips).filter(|c| c.text.as_ref().is_some_and(|x| x.style == k)).count();
                Ok(json!({ "styles": seq.styles.iter().enumerate().map(|(k, s)| {
                    let mut v = serde_json::to_value(s).unwrap_or_default();
                    v["clips"] = json!(users(k));
                    v
                }).collect::<Vec<_>>(), "builtin_note": "styles appear with the first text clip; create=true makes one" }))
            }
            "style_set" => {
                self.rec.seq.ensure_styles();
                let name = string(p, "style").unwrap_or_default().to_string();
                let k = match self.rec.seq.style_named(&name) {
                    Some(k) => k,
                    None if boolean(p, "create") == Some(true) => {
                        let base = match string(p, "from") {
                            Some(f) => self.rec.seq.styles.get(self.rec.seq.style_named(f).ok_or_else(|| no(format!("no style {f}")))?).cloned().unwrap_or_default(),
                            None => model::TextStyle::default(),
                        };
                        self.rec.seq.styles.push(model::TextStyle { name: name.clone(), ..base });
                        self.rec.seq.styles.len() - 1
                    }
                    None => return Err(no(format!("no style {name} (create=true makes it)"))),
                };
                let mut v = serde_json::to_value(&self.rec.seq.styles[k]).unwrap_or_default();
                let mut place = None;
                if let Some(f) = p.get("fields").and_then(Value::as_object) {
                    for (key, val) in f {
                        if key == "place" {
                            place = model::PLACES.iter().position(|q| Some(q.0) == val.as_str());
                            if place.is_none() {
                                return Err(bad(format!("place: one of {}", model::PLACES.iter().map(|q| q.0).collect::<Vec<_>>().join(", "))));
                            }
                            continue;
                        }
                        if v.get(key).is_none() {
                            return Err(bad(format!("a style has no field '{key}'")));
                        }
                        v[key] = val.clone();
                    }
                }
                let mut st: model::TextStyle = serde_json::from_value(v).map_err(|e| bad(format!("style fields: {e}")))?;
                if let Some(i) = place {
                    st.place(i);
                }
                self.rec.seq.styles[k] = st.clone();
                self.changed(false);
                Ok(serde_json::to_value(&st).unwrap_or_default())
            }
            "srt_import" => {
                let track = match p.get("track") {
                    Some(_) => Some(track_of(p, "track")?),
                    None => None,
                };
                let style = match string(p, "style") {
                    Some(n) => {
                        self.rec.seq.ensure_styles();
                        Some(self.rec.seq.style_named(n).ok_or_else(|| no(format!("no style {n}")))?)
                    }
                    None => None,
                };
                let n = self.import_srt(&path_of(p, "path"), track, style).map_err(no)?;
                Ok(json!({ "imported": n }))
            }
            "srt_export" => {
                let track = match p.get("track") {
                    Some(_) => Some(track_of(p, "track")?),
                    None => None,
                };
                let n = self.export_srt(&path_of(p, "path"), track).map_err(no)?;
                Ok(json!({ "exported": n }))
            }
            "render_presets" => {
                let raster = self.rec.seq.raster.unwrap_or((1920, 1080));
                let best = crate::render::preset_for(raster);
                Ok(json!({ "presets": crate::render::PRESETS.iter().map(|q| json!({ "id": q.id, "name": q.name, "width": q.width, "height": q.height, "bitrate": q.bitrate })).collect::<Vec<_>>(), "matches_sequence": best.id }))
            }
            "render_start" => {
                if self.rec.seq.is_empty() {
                    return Err(no("nothing to render: the sequence is empty"));
                }
                if self.export.busy() {
                    return Err(no("a render is already running (render_status)"));
                }
                let preset = crate::render::preset(string(p, "preset").unwrap_or_default()).ok_or_else(|| bad("unknown preset (render_presets lists them)"))?;
                let fit = if boolean(p, "fill") == Some(true) { crate::render::Fit::Fill } else { crate::render::Fit::Fit };
                let out = match &self.post {
                    Some(q) => q.out_path(preset.id),
                    None => {
                        string(p, "path").ok_or_else(|| no("not in a marketing post: give a 'path' (.mp4)"))?;
                        let path = path_of(p, "path");
                        crate::render::free_name(&path.with_extension("mp4"))
                    }
                };
                let job = self.render_job(preset, fit, boolean(p, "inout").unwrap_or(false), out.clone(), None);
                let frames = job.range.1 - job.range.0;
                self.start_render(job);
                Ok(json!({ "out": out, "frames": frames }))
            }
            "render_status" => Ok(match self.export.progress() {
                Some((name, done, total, fps)) => json!({ "running": true, "file": name, "done": done, "total": total, "fps": fps }),
                None => json!({ "running": false, "last": self.export.last }),
            }),
            "render_cancel" => {
                if !self.export.busy() {
                    return Err(no("no render running"));
                }
                self.export.cancel();
                Ok(json!({ "ok": true }))
            }
            "undo" | "redo" => {
                let before = self.rec.history.depth();
                self.undo_public_dir(method == "redo");
                Ok(json!({ "changed": self.rec.history.depth() != before, "undo": self.rec.history.depth().0, "redo": self.rec.history.depth().1 }))
            }
            "batch" => {
                let calls = p.get("calls").and_then(Value::as_array).cloned().unwrap_or_default();
                let mut results = Vec::new();
                for (k, c) in calls.iter().enumerate() {
                    let m = c.get("method").and_then(Value::as_str).unwrap_or_default();
                    let params = c.get("params").cloned().unwrap_or_else(|| json!({}));
                    if NOT_IN_BATCH.contains(&m) {
                        return Err(no(format!("call {k}: {m} cannot be batched")));
                    }
                    let Some(def) = api::method(m) else { return Err(ApiError { code: -32601, message: format!("call {k}: unknown method {m}") }) };
                    api::validate(&def, &params).map_err(|e| ApiError { code: e.code, message: format!("call {k}: {}", e.message) })?;
                    match self.api_call(m, &params) {
                        Ok(v) => results.push(v),
                        Err(e) => return Err(ApiError { code: e.code, message: format!("call {k} ({m}): {}", e.message) }),
                    }
                }
                Ok(json!({ "results": results }))
            }
            m => Err(ApiError { code: -32601, message: format!("{m} is in the table but not implemented") }),
        }
    }

    fn api_targets(&self) -> Value {
        let n = |t: Option<usize>| t.map(|t| TRACKS[t].0);
        json!({ "video": n(self.rec.targets.video), "audio": n(self.rec.targets.audio) })
    }

    fn api_selection(&self) -> Value {
        let n = |t: usize| TRACKS[t].0;
        match &self.rec.sel {
            Selection::None => json!(null),
            Selection::Clip { track, start } => json!({ "clips": [{ "track": n(*track), "start": start }] }),
            Selection::Clips(v) => json!({ "clips": v.iter().map(|(t, s)| json!({ "track": n(*t), "start": s })).collect::<Vec<_>>() }),
            Selection::Edit { track, at, side } => json!({ "edit": { "track": n(*track), "at": at, "side": side_name(*side) } }),
        }
    }

    fn api_info(&self) -> Value {
        let r = &self.rec;
        let marks = |m: model::Marks| json!({ "in": m.inp, "out": m.out });
        json!({
            "protocol": api::VERSION,
            "project": { "path": self.project.path, "name": self.project.name(), "unsaved": self.project.dirty, "offline_media": self.project.missing.len() },
            "post": self.post.as_ref().map(|q| json!({ "title": q.title, "dir": q.dir(), "frame": q.frame })),
            "sequence": { "fps": [r.seq.fps.num, r.seq.fps.den], "fps_f": r.seq.fps.as_f64(), "raster": r.seq.raster, "duration": r.seq.duration(), "clips": r.seq.tracks.iter().map(|t| t.clips.len()).sum::<usize>() },
            "record": {
                "playhead": r.transport.frame(),
                "playing": r.transport.playing(),
                "marks": marks(r.marks),
                "targets": self.api_targets(),
                "tool": tool_name(r.tool),
                "ripple_trims": r.ripple,
                "snapping": r.snap,
                "selection": self.api_selection(),
                "undo": r.history.depth().0,
                "redo": r.history.depth().1,
            },
            "source": self.source.as_ref().map(|c| json!({
                "path": c.original,
                "frames": c.info.frames,
                "fps": c.info.fps(),
                "width": c.info.width,
                "height": c.info.height,
                "has_audio": c.info.has_audio,
                "playhead": c.transport.frame(),
                "playing": c.transport.playing(),
                "marks": marks(c.marks),
            })),
            "focus": format!("{:?}", self.focus).to_lowercase(),
            "mode": mode_name(self.mode),
            "rendering": self.export.busy(),
            "message": self.last_notice(),
        })
    }

    fn api_timeline(&self) -> Value {
        let s = &self.rec.seq;
        json!({
            "fps": [s.fps.num, s.fps.den],
            "raster": s.raster,
            "duration": s.duration(),
            "media": s.media.iter().map(|m| json!({ "path": m.path, "name": m.name, "frames": m.frames, "fps": [m.fps.num, m.fps.den], "has_audio": m.has_audio, "width": m.width, "height": m.height })).collect::<Vec<_>>(),
            "tracks": s.tracks.iter().map(|t| json!({
                "name": t.name,
                "kind": if t.kind == model::Kind::Video { "video" } else { "audio" },
                "mute": t.mute,
                "solo": t.solo,
                "gain_db": t.gain_db,
                "clips": t.clips.iter().map(|c| {
                    let (hb, ha) = s.handles(c);
                    json!({
                        "start": c.start, "len": c.len, "end": c.end(), "src_in": c.src_in, "speed": c.speed,
                        "media": c.media, "name": s.media[c.media].name, "link": c.link,
                        "reframe": { "fill": c.reframe.fill, "zoom": c.reframe.zoom, "x": c.reframe.x, "y": c.reframe.y },
                        "handles": [hb, ha],
                        "text": c.text.as_ref().map(|t| json!({ "text": t.text, "style": s.styles.get(t.style).map(|x| x.name.clone()) })),
                    })
                }).collect::<Vec<_>>(),
                "dissolves": t.dissolves.iter().map(|d| json!({ "at": d.at, "before": d.before, "after": d.after })).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    }

    /// `viewer_capture`: compose the frame on a worker and answer from it.
    fn api_capture(&mut self, call: Call) {
        let p = call.params.clone();
        let max = int(&p, "max_size").unwrap_or(768).clamp(16, 4096) as u32;
        let source = string(&p, "viewer") == Some("source");
        let job = if source {
            let Some(c) = &self.source else {
                call.answer(Err(no("no clip in the source viewer")));
                return;
            };
            let fps = model::Fps::new(c.info.fps_num, c.info.fps_den);
            let mut seq = model::Sequence::new(fps);
            seq.raster = Some((c.info.width, c.info.height));
            let m = seq.add_media(model::Media { path: c.original.clone(), name: String::new(), frames: c.info.frames as i64, fps, has_audio: false, width: c.info.width, height: c.info.height, generated: false, audio_only: false });
            seq.overwrite(model::Source::new(m, 0, c.info.frames as i64), 0, model::Targets { video: Some(model::V1), audio: None });
            let read = vec![c.server.source().info().path.clone()];
            let t = int(&p, "frame").unwrap_or(c.transport.frame() as i64);
            (seq, read, t)
        } else {
            let seq = self.rec.seq.clone();
            let read = self.read_paths(&seq);
            let t = int(&p, "frame").unwrap_or(self.rec.transport.frame() as i64);
            (seq, read, t)
        };
        let _ = std::thread::Builder::new().name("api-capture".into()).spawn(move || {
            let (seq, read, t) = job;
            let r = crate::render::still(&seq, &read, t, max)
                .map(|(png, w, h)| json!({ "png_base64": api::b64(&png), "width": w, "height": h, "frame": t }))
                .map_err(|e| no(format!("capture: {e:#}")));
            call.answer(r);
        });
    }
}

/// A path param, with a leading `~` expanded.
fn rating_name(r: Option<crate::bin::Rating>) -> Option<&'static str> {
    r.map(|r| match r {
        crate::bin::Rating::Select => "select",
        crate::bin::Rating::Reject => "reject",
    })
}

fn mode_name(m: crate::editor::Mode) -> &'static str {
    match m {
        crate::editor::Mode::Edit => "edit",
        crate::editor::Mode::Selects => "selects",
    }
}

fn filter_of(p: &Value) -> Result<crate::bin::Filter, ApiError> {
    let s = string(p, "filter").unwrap_or("all");
    crate::bin::Filter::ALL.into_iter().find(|f| f.name() == s).ok_or_else(|| bad(format!("filter {s}: all, selects, unmarked or rejects")))
}

fn path_of(p: &Value, k: &str) -> PathBuf {
    imfram::filebrowser::expand_home(string(p, k).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every method in the table has an arm here (an empty call may be
    /// refused for its params, never as "not implemented").
    #[test]
    fn the_app_implements_every_method() {
        let mut ed = Editor::new(true);
        for m in api::methods() {
            if m.name == "viewer_capture" {
                continue; // answered on a worker (api_capture)
            }
            if let Err(e) = ed.api_recorded(m.name, &json!({})) {
                assert_ne!(e.code, -32601, "{}: {}", m.name, e.message);
            }
        }
    }

    /// Selects through the API: verdicts, the filtered list, subclips
    /// added, listed, laid down in one undo step, and removed.
    #[test]
    fn selects_tools() {
        let dir = std::env::temp_dir().join(format!("scrub-api-selects-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..10 {
            let f = std::fs::File::create(dir.join(format!("take.{:04}.png", 1001 + i))).unwrap();
            let mut e = png::Encoder::new(std::io::BufWriter::new(f), 8, 8);
            e.set_color(png::ColorType::Rgba);
            e.write_header().unwrap().write_image_data(&[128u8; 8 * 8 * 4]).unwrap();
        }
        let clip = dir.join("take.1001.png");
        let mut ed = Editor::new(true);
        let call = |ed: &mut Editor, m: &str, p: Value| ed.api_recorded(m, &p);
        assert_eq!(call(&mut ed, "mode_set", json!({ "mode": "selects" })).unwrap()["mode"], "selects");
        call(&mut ed, "bin_add", json!({ "path": clip })).unwrap();
        call(&mut ed, "clip_rate", json!({ "path": clip, "rating": "reject" })).unwrap();
        let l = call(&mut ed, "selects_list", json!({ "filter": "rejects" })).unwrap();
        assert_eq!(l["clips"].as_array().unwrap().len(), 1, "{l}");
        assert_eq!(l["counts"]["selects"], 0);
        call(&mut ed, "subclip_add", json!({ "path": clip, "in": 2, "out": 4 })).unwrap();
        call(&mut ed, "subclip_add", json!({ "in": 6, "out": 9 })).unwrap();
        assert!(call(&mut ed, "subclip_add", json!({ "in": 6, "out": 9 })).is_err(), "a repeat is refused");
        assert!(call(&mut ed, "subclip_add", json!({ "in": 6, "out": 99 })).is_err(), "past the end is refused");
        let s = call(&mut ed, "subclip_list", json!({})).unwrap();
        assert_eq!(s["subclips"].as_array().unwrap().len(), 2, "{s}");
        assert_eq!(s["subclips"][1]["frames"], 4);
        let before = ed.rec.history.depth().0;
        let r = call(&mut ed, "subclips_place", json!({})).unwrap();
        assert_eq!(r["placed"], 2, "{r}");
        assert_eq!(ed.rec.seq.tracks[model::V1].clips.len(), 2);
        assert_eq!(ed.rec.history.depth().0, before + 1, "one undo step");
        call(&mut ed, "subclip_remove", json!({ "path": clip, "in": 2, "out": 4 })).unwrap();
        assert_eq!(call(&mut ed, "subclip_list", json!({})).unwrap()["subclips"].as_array().unwrap().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A failed call puts the sequence back and makes no undo step; a good
    /// one is exactly one step.
    #[test]
    fn a_call_is_one_undo_step_and_a_failure_none() {
        let mut ed = Editor::new(true);
        let m = ed.rec.seq.add_media(model::Media { path: "/x.mov".into(), name: "x".into(), frames: 500, fps: model::Fps::new(24, 1), has_audio: true, width: 64, height: 36, generated: false, audio_only: false });
        ed.rec.seq.overwrite(model::Source::new(m, 0, 20), 0, model::Targets::default());
        ed.rec.history.reset(&ed.rec.seq);
        let r = ed.api_recorded("batch", &json!({ "calls": [
            { "method": "razor", "params": { "at": 5 } },
            { "method": "razor", "params": { "at": 10 } },
        ] }));
        assert!(r.is_ok(), "{r:?}");
        assert_eq!(ed.rec.seq.tracks[model::V1].clips.len(), 3);
        assert_eq!(ed.rec.history.depth().0, 1);
        let before = ed.rec.seq.clone();
        let r = ed.api_recorded("batch", &json!({ "calls": [
            { "method": "razor", "params": { "at": 15 } },
            { "method": "dissolve_remove", "params": { "track": "V1", "at": 5 } },
        ] }));
        assert!(r.unwrap_err().message.contains("call 1"));
        assert_eq!(ed.rec.seq, before);
        assert_eq!(ed.rec.history.depth().0, 1);
        ed.api_recorded("undo", &json!({})).unwrap();
        assert_eq!(ed.rec.seq.tracks[model::V1].clips.len(), 1);
    }
}
