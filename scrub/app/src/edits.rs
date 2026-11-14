//! Step 3's edits, driven by keys: select the clip or the edit point under
//! the playhead, delete / ripple delete, trim the selected edit by frames
//! (out, in or roll; ripple or not), extend it to the playhead, slip the
//! selected clip, set its speed, fit to fill, and dissolves.
//!
//! The selection lives on the record side (`Record::sel`). A clip is held
//! by its track and start (indices move under edits); an edit point by its
//! track, cut and side. Trims are bursts: holding a trim key makes one undo
//! step once it settles.

use crate::editor::Editor;
use crate::timelineui::Selection;
use model::{Layer, Side, VIDEO_TRACKS};
use std::time::Instant;

/// How long the slip monitor and band stay up after a slip key.
pub const SLIP_SHOW_SECS: f32 = 1.5;

/// Default dissolve length: one second at 24 fps.
pub const DISSOLVE_FRAMES: i64 = 24;

impl Editor {
    pub(crate) fn say(&mut self, msg: impl Into<String>) {
        self.notice = Some((msg.into(), Instant::now()));
    }

    /// The edit changed the sequence: one undo step now (`burst` false), or
    /// when the burst settles (trim keys held).
    pub(crate) fn changed(&mut self, burst: bool) {
        let r = &mut self.rec;
        if self.api_quiet {
            // A scripting call makes its own undo step (`apiserve`).
            r.fit_transport();
            return;
        }
        r.history.touched();
        if !burst {
            r.history.commit(&r.seq);
        }
        r.fit_transport();
    }

    /// The tracks to look on for a clip at the playhead: the targeted video
    /// track first, then the others top down, then the audio tracks.
    fn track_order(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.rec.targets.video.into_iter().collect();
        for &t in VIDEO_TRACKS.iter().rev() {
            if !v.contains(&t) {
                v.push(t);
            }
        }
        v.extend(model::AUDIO_TRACKS);
        v
    }

    /// D: select the clip under the playhead (targeted track first). Again:
    /// the next track down that has one.
    pub(crate) fn select_clip(&mut self, cycle: bool) {
        let t = self.rec.transport.frame() as i64;
        let order = self.track_order();
        let have: Vec<usize> = order.into_iter().filter(|&tr| self.rec.seq.clip_at(tr, t).is_some()).collect();
        if have.is_empty() {
            self.rec.sel = Selection::None;
            self.say("no clip under the playhead");
            return;
        }
        let next = match self.rec.sel {
            Selection::Clip { track, .. } if cycle => {
                let i = have.iter().position(|&x| x == track).map_or(0, |i| (i + 1) % have.len());
                have[i]
            }
            _ => have[0],
        };
        let i = self.rec.seq.clip_at(next, t).expect("filtered above");
        let start = self.rec.seq.tracks[next].clips[i].start;
        self.rec.sel = Selection::Clip { track: next, start };
    }

    /// The selected clip's (track, index), or the clip under the playhead
    /// on the targeted track when nothing is selected.
    pub(crate) fn clip_target(&self) -> Option<(usize, usize)> {
        let seq = &self.rec.seq;
        match &self.rec.sel {
            Selection::Clip { track, start } => seq.tracks[*track].clips.iter().position(|c| c.start == *start).map(|i| (*track, i)),
            Selection::Clips(v) => v.first().and_then(|&(t, st)| seq.tracks[t].clips.iter().position(|c| c.start == st).map(|i| (t, i))),
            _ => {
                let t = self.rec.transport.frame() as i64;
                self.track_order().into_iter().find_map(|tr| seq.clip_at(tr, t).map(|i| (tr, i)))
            }
        }
    }

    /// Delete: lift the selected clip (and its audio). Shift: ripple delete.
    pub(crate) fn delete_clip(&mut self, ripple: bool) {
        // Several selected: lift them all (a ripple delete takes one clip).
        if let (Selection::Clips(v), false) = (&self.rec.sel, ripple) {
            for (t, st) in v.clone() {
                if let Some(i) = self.rec.seq.tracks[t].clips.iter().position(|c| c.start == st) {
                    self.rec.seq.lift_clip(t, i);
                }
            }
            self.rec.sel = Selection::None;
            self.changed(false);
            return;
        }
        let Some((t, i)) = self.clip_target() else {
            self.say("select a clip first (D)");
            return;
        };
        if ripple {
            if let Err(e) = self.rec.seq.ripple_delete(t, i) {
                self.say(e.0);
                return;
            }
        } else {
            self.rec.seq.lift_clip(t, i);
        }
        self.rec.sel = Selection::None;
        self.changed(false);
    }

    /// T: select the cut nearest the playhead on the targeted track (roll).
    /// Again: cycle roll -> out -> in.
    pub(crate) fn select_edit(&mut self) {
        if let Selection::Edit { track, at, side } = self.rec.sel {
            let side = match side {
                Side::Roll => Side::Out,
                Side::Out => Side::In,
                Side::In => Side::Roll,
            };
            self.rec.sel = Selection::Edit { track, at, side };
            return;
        }
        let t = self.rec.transport.frame() as i64;
        let seq = &self.rec.seq;
        let best = self
            .track_order()
            .into_iter()
            .filter_map(|tr| seq.nearest_cut(tr, t).map(|at| (tr, at)))
            .min_by_key(|&(_, at)| (at - t).abs());
        match best {
            Some((track, at)) => {
                self.rec.sel = Selection::Edit { track, at, side: Side::Roll };
                self.rec.transport.stop();
                self.rec.transport.seek(at as u64);
            }
            None => self.say("no edit points"),
        }
    }

    /// Trim the selected edit by `d` frames (alt+arrows).
    pub(crate) fn trim_by(&mut self, d: i64) {
        let Selection::Edit { track, at, side } = self.rec.sel else {
            self.say("select an edit point first (G)");
            return;
        };
        let ripple = self.rec.ripple && side != Side::Roll;
        match self.rec.seq.trim(track, at, side, d, ripple) {
            Ok(0) => self.say("at the limit: no more handle or room"),
            Ok(moved) => {
                // The selection follows the cut it trimmed.
                // (a ripple in-trim keeps the cut where it is; every other
                // trim moves it by what it moved)
                let at2 = if side == Side::In && ripple { at } else { at + moved };
                self.rec.sel = Selection::Edit { track, at: at2, side };
                self.rec.transport.seek(at2.max(0) as u64);
                self.changed(true);
                if moved != d {
                    self.say("stopped at the limit");
                }
            }
            Err(e) => self.say(e.0),
        }
    }

    /// E: move the selected edit to the playhead.
    pub(crate) fn extend_edit(&mut self) {
        let Selection::Edit { track, at, side } = self.rec.sel else {
            self.say("select an edit point first (G)");
            return;
        };
        let to = self.rec.transport.frame() as i64;
        let ripple = self.rec.ripple && side != Side::Roll;
        match self.rec.seq.extend(track, at, side, to, ripple) {
            Ok(moved) => {
                self.rec.sel = Selection::Edit { track, at: at + moved, side };
                self.changed(false);
                if at + moved != to {
                    self.say("stopped at the limit");
                }
            }
            Err(e) => self.say(e.0),
        }
    }

    /// Slip the selected clip by `d` source frames (alt+, / alt+.).
    pub(crate) fn slip_by(&mut self, d: i64) {
        let Some((t, i)) = self.clip_target() else {
            self.say("select a clip first (D)");
            return;
        };
        let moved = self.rec.seq.slip(t, i, d);
        if moved == 0 {
            self.say("no more source to slip into");
            return;
        }
        let start = self.rec.seq.tracks[t].clips[i].start;
        self.rec.sel = Selection::Clip { track: t, start };
        self.slip_view = Some((t, start, Instant::now()));
        self.changed(true);
    }

    /// The clip being slipped (a slip drag, or slip keys in the last 1.5 s).
    pub(crate) fn slipping(&self) -> Option<(usize, usize)> {
        let (t, st, at) = self.slip_view?;
        if at.elapsed().as_secs_f32() > SLIP_SHOW_SECS && !self.dragging() {
            return None;
        }
        // Only while that clip is still the selection.
        if !matches!(self.rec.sel, Selection::Clip { track, start } if (track, start) == (t, st)) {
            return None;
        }
        self.rec.seq.tracks[t].clips.iter().position(|c| c.start == st).map(|i| (t, i))
    }

    /// The viewers' words while they are a two-up monitor.
    pub(crate) fn monitor_words(&self) -> (&'static str, &'static str) {
        if self.slipping().is_some() {
            ("slip: the clip's first frame", "slip: the clip's last frame")
        } else {
            ("trim: outgoing out", "trim: incoming in")
        }
    }

    /// The handles band for a clip being slipped (timeline overlay).
    pub(crate) fn slip_band(&self) -> Option<(usize, i64, i64, i64, i64)> {
        let (t, i) = self.slipping()?;
        let c = &self.rec.seq.tracks[t].clips[i];
        let (h0, h1) = self.rec.seq.handles(c);
        Some((t, c.start - h0, c.start, c.end(), c.end() + h1))
    }

    /// Change how the selected clip's picture sits in the raster (reframe
    /// keys: F fit / fill, alt+F reset, [ ] zoom, cmd+arrows move). Held
    /// keys make one undo step once they settle.
    pub(crate) fn reframe_with(&mut self, f: impl FnOnce(&mut model::Reframe)) {
        let Some((t, i)) = self.clip_target() else {
            self.say("select a clip first (D)");
            return;
        };
        let mut r = self.rec.seq.tracks[t].clips[i].reframe;
        f(&mut r);
        r.zoom = r.zoom.clamp(0.05, 20.0);
        r.x = r.x.clamp(-2.0, 2.0);
        r.y = r.y.clamp(-2.0, 2.0);
        let start = self.rec.seq.tracks[t].clips[i].start;
        self.rec.seq.set_reframe(t, i, r);
        self.rec.sel = Selection::Clip { track: t, start };
        self.changed(true);
        self.say(reframe_words(&r));
    }

    /// Every video clip reframed alike, and optionally a new raster (checks).
    pub fn reframe_all(&mut self, raster: Option<(u32, u32)>, r: model::Reframe) {
        if raster.is_some() {
            self.rec.seq.raster = raster;
        }
        for &t in VIDEO_TRACKS.iter() {
            for i in 0..self.rec.seq.tracks[t].clips.len() {
                self.rec.seq.set_reframe(t, i, r);
            }
        }
        self.changed(false);
    }

    /// The reframe of the selected clip, in words (for the HUD), when not plain.
    pub(crate) fn reframe_line(&self) -> Option<String> {
        let (t, i) = self.clip_target()?;
        let r = self.rec.seq.tracks[t].clips[i].reframe;
        (!r.is_default()).then(|| reframe_words(&r))
    }

    /// cmd+R: open the speed dialog for the selected clip.
    pub(crate) fn ask_speed(&mut self) {
        match self.clip_target() {
            Some((t, i)) => {
                self.speed_dialog = Some(self.rec.seq.tracks[t].clips[i].speed);
                self.speed_dialog_focus = true;
            }
            None => self.say("select a clip first (D)"),
        }
    }

    pub(crate) fn apply_speed(&mut self, speed: f64) {
        let Some((t, i)) = self.clip_target() else { return };
        if speed == 0.0 || !speed.is_finite() {
            self.say("speed must not be 0");
            return;
        }
        let start = self.rec.seq.tracks[t].clips[i].start;
        let len = self.rec.seq.set_speed(t, i, speed);
        self.rec.sel = Selection::Clip { track: t, start };
        self.changed(false);
        self.say(format!("speed {speed}x: {len} frames"));
    }

    /// alt+R: conform the selected clip (one source frame per record frame)
    /// or put it back to real time. Its length follows; linked clips too.
    pub(crate) fn toggle_conform(&mut self) {
        let Some((t, i)) = self.clip_target() else {
            self.say("select a clip first (D)");
            return;
        };
        let c = &self.rec.seq.tracks[t].clips[i];
        let (start, on) = (c.start, !c.conform);
        let (src, seq) = (self.rec.seq.media[c.media].fps, self.rec.seq.fps);
        let len = self.rec.seq.set_conform(t, i, on);
        self.rec.sel = Selection::Clip { track: t, start };
        self.changed(false);
        let rates = format!("{:.3} fps in {:.3}", src.as_f64(), seq.as_f64());
        self.say(match (on, src.same(seq)) {
            (_, true) => format!("conform {}: same rate, no change", if on { "on" } else { "off" }),
            (true, false) => format!("conformed ({rates}): every frame, {len} frames, sound muted"),
            (false, false) => format!("real time ({rates}): {len} frames"),
        });
    }

    /// shift+.: fit to fill. The source in..out stretched over the record
    /// in..out (all four marks).
    pub(crate) fn fit_to_fill(&mut self) {
        let Some(c) = &self.source else { return };
        let (Some(si), Some(so)) = (c.marks.inp, c.marks.out) else {
            self.say("fit to fill needs a source in and out");
            return;
        };
        let (Some(ri), Some(ro)) = (self.rec.marks.inp, self.rec.marks.out) else {
            self.say("fit to fill needs a record in and out");
            return;
        };
        let media = self.media_of_source();
        let targets = self.rec.targets;
        self.rec.seq.overwrite_fit(media, si, so - si + 1, ri, ro - ri + 1, targets);
        self.rec.marks.clear();
        self.changed(false);
        self.rec.transport.seek((ro + 1) as u64);
        let speed = self.rec.seq.tracks.iter().flat_map(|t| t.clips.iter()).find(|x| x.start == ri).map_or(1.0, |x| x.speed);
        self.say(format!("fit to fill at {speed:.3}x"));
    }

    /// The cut a dissolve key acts on: the selected edit point; two
    /// selected clips that meet on one track (the cut between them); else
    /// the nearest cut on the targeted video track. `Err`: why there is
    /// none (said on the HUD).
    fn dissolve_cut(&self) -> Result<(usize, i64), String> {
        let seq = &self.rec.seq;
        match &self.rec.sel {
            Selection::Edit { track, at, .. } => Ok((*track, *at)),
            Selection::Clips(v) => {
                // Linked sound selected with its picture counts once: the
                // picture decides.
                let video: Vec<(usize, i64)> = v.iter().copied().filter(|(t, _)| VIDEO_TRACKS.contains(t)).collect();
                let pick = if video.len() >= 2 { video } else { v.clone() };
                if pick.len() != 2 {
                    return Err(format!("{} clips selected: a dissolve goes between two clips that meet", pick.len()));
                }
                let ((ta, sa), (tb, sb)) = (pick[0], pick[1]);
                if ta != tb {
                    return Err(format!("the two clips are on {} and {}: a dissolve needs them on one track", seq.tracks[ta].name, seq.tracks[tb].name));
                }
                let find = |s: i64| seq.tracks[ta].clips.iter().find(|c| c.start == s);
                let (Some(a), Some(b)) = (find(sa), find(sb)) else { return Err("the selection is out of date".into()) };
                let (a, b) = if a.start < b.start { (a, b) } else { (b, a) };
                if a.end() != b.start {
                    return Err(format!("the two clips don't meet ({} frames apart): a dissolve needs them back to back", b.start - a.end()));
                }
                Ok((ta, b.start))
            }
            _ => {
                let t = self.rec.transport.frame() as i64;
                let tr = self.rec.targets.video.unwrap_or(model::V1);
                seq.nearest_cut(tr, t).map(|at| (tr, at)).ok_or_else(|| "no cut to dissolve".to_string())
            }
        }
    }

    /// cmd+D: a default dissolve at the cut (or the length given).
    pub(crate) fn add_dissolve(&mut self, len: Option<i64>) {
        match self.dissolve_cut() {
            Ok((track, at)) => self.dissolve_at_cut(track, at, len),
            Err(e) => self.say(e),
        }
    }

    /// A dissolve on the cut at `at` on `track` (default length, or `len`
    /// while resizing), saying how it went.
    pub(crate) fn dissolve_at_cut(&mut self, track: usize, at: i64, len: Option<i64>) {
        let want = len.unwrap_or(DISSOLVE_FRAMES).max(1);
        match self.rec.seq.add_dissolve(track, at, want) {
            Ok(d) => {
                self.changed(len.is_some());
                if !matches!(self.rec.sel, Selection::Edit { track: t, at: a, .. } if (t, a) == (track, at)) {
                    self.rec.sel = Selection::Edit { track, at, side: Side::Roll };
                }
                if d.len() < want {
                    self.say(format!("dissolve clamped to {} frames ({} before, {} after): not enough handle", d.len(), d.before, d.after));
                } else {
                    self.say(format!("dissolve {} frames on the {} cut", d.len(), self.rec.seq.tracks[track].name));
                }
            }
            Err(e) => self.say(e.0),
        }
    }

    /// cmd+K: cut at the record playhead on the targeted tracks (each with
    /// its linked group); `all`: every track.
    pub(crate) fn razor_playhead(&mut self, all: bool) {
        let at = self.rec.transport.frame() as i64;
        if all {
            self.razor_at(None, at, true);
            return;
        }
        let tracks: Vec<usize> = self.rec.targets.video.into_iter().chain(self.rec.targets.audio).collect();
        if tracks.is_empty() {
            self.say("no tracks targeted (1-7): nothing to cut");
            return;
        }
        let n = self.rec.seq.razor_tracks(&tracks, at);
        if n == 0 {
            self.say("nothing on the targeted tracks at the playhead");
            return;
        }
        self.rec.sel = Selection::None;
        self.changed(false);
        let tc = crate::editor::timecode(at.max(0) as u64, self.rec.seq.fps.as_f64());
        self.say(format!("cut at {tc}"));
    }

    /// alt+= / alt+-: lengthen or shorten the dissolve at the cut.
    pub(crate) fn dissolve_by(&mut self, d: i64) {
        let Ok((track, at)) = self.dissolve_cut() else { return };
        let Some(cur) = self.rec.seq.tracks[track].dissolve_at(at).copied() else {
            self.say("no dissolve here (cmd+D adds one)");
            return;
        };
        self.dissolve_at_cut(track, at, Some((cur.len() + d).max(1)));
    }

    pub(crate) fn remove_dissolve(&mut self) {
        match self.dissolve_cut() {
            Ok((track, at)) => {
                if self.rec.seq.remove_dissolve(track, at) {
                    self.changed(false);
                    self.say("dissolve removed");
                } else {
                    self.say("no dissolve on that cut");
                }
            }
            Err(e) => self.say(e),
        }
    }

    /// While an edit point is selected the viewers become a trim monitor:
    /// the outgoing clip's last frame left, the incoming clip's first right.
    pub(crate) fn trim_frames(&self) -> Option<(Option<Layer>, Option<Layer>)> {
        if self.rec.transport.playing() {
            return None;
        }
        // A slip shows the clip's first frame left and its last right.
        if let Some((t, i)) = self.slipping() {
            let seq = &self.rec.seq;
            let c = &seq.tracks[t].clips[i];
            let at = |f: i64| Layer { media: c.media, frame: seq.source_frame(c, f), reframe: c.reframe };
            return Some((Some(at(c.start)), Some(at(c.end() - 1))));
        }
        let Selection::Edit { track, at, .. } = self.rec.sel else { return None };
        let seq = &self.rec.seq;
        let t = &seq.tracks[track];
        let (a, b) = t.cut(at);
        let la = a.map(|i| {
            let c = &t.clips[i];
            Layer { media: c.media, frame: seq.source_frame(c, c.end() - 1), reframe: c.reframe }
        });
        let lb = b.map(|i| {
            let c = &t.clips[i];
            Layer { media: c.media, frame: seq.source_frame(c, c.start), reframe: c.reframe }
        });
        Some((la, lb))
    }

    /// The media index of the source clip in the sequence (added if new).
    pub(crate) fn media_of_source(&mut self) -> usize {
        let c = self.source.as_ref().expect("checked by callers");
        let fps = model::Fps::new(c.info.fps_num, c.info.fps_den);
        if self.rec.seq.is_empty() {
            // The first edit sets the rate, and the raster unless chosen.
            self.rec.seq.fps = fps;
            if self.rec.seq.raster.is_none() {
                self.rec.seq.raster = Some((c.info.width, c.info.height));
            }
        }
        let name = c.original.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        self.rec.seq.add_media(model::Media {
            path: c.original.clone(),
            name,
            frames: c.info.frames as i64,
            fps,
            has_audio: c.info.has_audio,
            width: c.info.width,
            height: c.info.height,
            generated: false,
        })
    }
}

fn reframe_words(r: &model::Reframe) -> String {
    format!("reframe: {} {:.0}%  x {:+.0}%  y {:+.0}%", if r.fill { "fill" } else { "fit" }, r.zoom * 100.0, r.x * 100.0, r.y * 100.0)

}

impl Editor {
    /// A track header's mute / solo / gain. Gain drags are one undo step
    /// when they end.
    pub(crate) fn track_op(&mut self, t: usize, op: crate::timelineui::TrackOp) {
        use crate::timelineui::TrackOp;
        let tr = &mut self.rec.seq.tracks[t];
        let burst = match op {
            TrackOp::Mute => {
                tr.mute = !tr.mute;
                false
            }
            TrackOp::Solo => {
                tr.solo = !tr.solo;
                false
            }
            TrackOp::Gain(db, done) => {
                tr.gain_db = (db * 10.0).round() / 10.0;
                !done
            }
        };
        let m = format!("{}: {}{}gain {:+.1} dB", tr.name, if tr.mute { "muted, " } else { "" }, if tr.solo { "solo, " } else { "" }, tr.gain_db);
        self.changed(burst);
        self.say(m);
    }
}

