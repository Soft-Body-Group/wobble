//! The clip edits: delete, ripple delete, trims at a cut (out, in, roll;
//! ripple or not), extend to the playhead, slip, speed, dissolves. Each
//! applies to a clip's whole linked GROUP (its video and its audio), and
//! clamps to what the media and the neighbours allow rather than failing
//! half way: a trim key held against a limit simply stops there.

use crate::{Clip, Dissolve, Sequence, A1, A3, AUDIO_TRACKS, M, V3, VIDEO_TRACKS};

/// Which side of a cut a trim moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The outgoing clip's out point.
    Out,
    /// The incoming clip's in point.
    In,
    /// Both: the cut moves, total length stays.
    Roll,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrimError(pub String);

impl std::fmt::Display for TrimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

fn err<T>(s: &str) -> Result<T, TrimError> {
    Err(TrimError(s.to_string()))
}

impl Sequence {
    /// The clip on `track` at `t`.
    pub fn clip_at(&self, track: usize, t: i64) -> Option<usize> {
        self.tracks[track].index_at(t)
    }

    /// A clip and every clip linked to it that sits in step with it (same
    /// start and length), as (track, index).
    pub fn group(&self, track: usize, idx: usize) -> Vec<(usize, usize)> {
        let c = &self.tracks[track].clips[idx];
        let mut v = vec![(track, idx)];
        for (ti, t) in self.tracks.iter().enumerate() {
            if ti == track {
                continue;
            }
            if let Some(i) = t.clips.iter().position(|o| o.link == c.link && o.start == c.start && o.len == c.len) {
                v.push((ti, i));
            }
        }
        v
    }

    fn clip(&self, at: (usize, usize)) -> &Clip {
        &self.tracks[at.0].clips[at.1]
    }

    /// Remove a clip (and its linked group), leaving a gap.
    pub fn lift_clip(&mut self, track: usize, idx: usize) {
        let mut g = self.group(track, idx);
        g.sort_by(|a, b| b.1.cmp(&a.1));
        for (t, i) in g {
            self.tracks[t].clips.remove(i);
        }
        self.tidy();
    }

    /// Remove a clip (and its group) and close the gap on every track.
    /// Refused when another track has material under the clip: closing the
    /// gap would pull it out of sync (lift instead, or clear that first).
    pub fn ripple_delete(&mut self, track: usize, idx: usize) -> Result<(), TrimError> {
        let g = self.group(track, idx);
        let (s, e) = (self.clip(g[0]).start, self.clip(g[0]).end());
        for (ti, t) in self.tracks.iter().enumerate() {
            if g.iter().any(|&(gt, _)| gt == ti) {
                continue;
            }
            if t.clips.iter().any(|c| c.start < e && c.end() > s) {
                return err(&format!("{} has material under this clip: lift it instead", t.name));
            }
        }
        self.lift_clip(track, idx);
        self.shift_from(e, s - e);
        self.tidy();
        Ok(())
    }

    /// A clip crosses `at` on a track outside `skip`: a ripple there would
    /// pull that track apart.
    fn crossing(&self, at: i64, skip: &[(usize, usize)]) -> Option<&'static str> {
        self.tracks.iter().enumerate().find_map(|(ti, t)| {
            let crosses = t.clips.iter().enumerate().any(|(i, c)| c.start < at && c.end() > at && !skip.contains(&(ti, i)));
            crosses.then_some(t.name)
        })
    }

    /// Trim the cut at `at` on `track` by `d` frames (positive: later).
    /// Out moves the outgoing clip's end, In the incoming clip's start,
    /// Roll both. `ripple`: the material after moves with the trim (every
    /// track), so nothing is overwritten and no gap opens. Returns the
    /// frames actually moved, after clamping to handles and neighbours.
    pub fn trim(&mut self, track: usize, at: i64, side: Side, d: i64, ripple: bool) -> Result<i64, TrimError> {
        let (a, b) = self.tracks[track].cut(at);
        let ga = a.map(|i| self.group(track, i)).unwrap_or_default();
        let gb = b.map(|i| self.group(track, i)).unwrap_or_default();
        let (lo, hi) = match side {
            Side::Out => {
                if ga.is_empty() {
                    return err("no clip ends at this cut");
                }
                let mut lo = i64::MIN;
                let mut hi = i64::MAX;
                for &g in &ga {
                    let c = self.clip(g);
                    lo = lo.max(-(c.len - 1));
                    hi = hi.min(self.handles(c).1);
                    if !ripple {
                        let next = self.tracks[g.0].clips.get(g.1 + 1).map_or(i64::MAX, |n| n.start - c.end());
                        hi = hi.min(next);
                    }
                }
                (lo, hi)
            }
            Side::In => {
                if gb.is_empty() {
                    return err("no clip starts at this cut");
                }
                let mut lo = i64::MIN;
                let mut hi = i64::MAX;
                for &g in &gb {
                    let c = self.clip(g);
                    hi = hi.min(c.len - 1);
                    lo = lo.max(-self.handles(c).0);
                    if !ripple {
                        let prev = if g.1 == 0 { c.start } else { c.start - self.tracks[g.0].clips[g.1 - 1].end() };
                        lo = lo.max(-prev);
                    }
                }
                (lo, hi)
            }
            Side::Roll => {
                if ga.is_empty() || gb.is_empty() {
                    return err("rolling needs a clip on both sides of the cut");
                }
                let mut lo = i64::MIN;
                let mut hi = i64::MAX;
                for &g in &ga {
                    let c = self.clip(g);
                    lo = lo.max(-(c.len - 1));
                    hi = hi.min(self.handles(c).1);
                }
                for &g in &gb {
                    let c = self.clip(g);
                    hi = hi.min(c.len - 1);
                    lo = lo.max(-self.handles(c).0);
                }
                (lo, hi)
            }
        };
        let d = d.clamp(lo.min(0), hi.max(0));
        if d == 0 {
            return Ok(0);
        }
        if ripple && side != Side::Roll {
            let moved = match side {
                Side::Out => &ga,
                _ => &gb,
            };
            if let Some(name) = self.crossing(at, moved) {
                return err(&format!("{name} has a clip across this cut: a ripple would pull it out of sync"));
            }
        }
        let (fps, media) = (self.fps, self.media.clone());
        match side {
            Side::Out => {
                for &g in &ga {
                    self.tracks[g.0].clips[g.1].len += d;
                }
                if ripple {
                    // Everything after the old out moves by the trim
                    // (the trimmed clips themselves start before it), and
                    // a dissolve on this cut with it.
                    self.shift_from(at, d);
                    for t in &mut self.tracks {
                        for x in &mut t.dissolves {
                            if x.at == at {
                                x.at += d;
                            }
                        }
                    }
                }
            }
            Side::In => {
                for &g in &gb {
                    let c = &mut self.tracks[g.0].clips[g.1];
                    c.src_in += c.src_offset(d, fps, media[c.media].fps);
                    c.len -= d;
                    if !ripple {
                        c.start += d;
                    }
                }
                if ripple {
                    // The head is trimmed in place and everything after the
                    // clip closes up (or opens out) behind it.
                    let end = self.clip(gb[0]).end() + d;
                    let skip: Vec<(usize, usize)> = gb.clone();
                    for (ti, t) in self.tracks.iter_mut().enumerate() {
                        for (i, c) in t.clips.iter_mut().enumerate() {
                            if c.start >= end && !skip.contains(&(ti, i)) {
                                c.start -= d;
                            }
                        }
                        for x in &mut t.dissolves {
                            if x.at >= end {
                                x.at -= d;
                            }
                        }
                    }
                }
            }
            Side::Roll => {
                for &g in &ga {
                    self.tracks[g.0].clips[g.1].len += d;
                }
                for &g in &gb {
                    let c = &mut self.tracks[g.0].clips[g.1];
                    c.src_in += c.src_offset(d, fps, media[c.media].fps);
                    c.start += d;
                    c.len -= d;
                }
                // The dissolve rides with its cut.
                for t in &mut self.tracks {
                    for x in &mut t.dissolves {
                        if x.at == at {
                            x.at += d;
                        }
                    }
                }
            }
        }
        for t in &mut self.tracks {
            t.clips.sort_by_key(|c| c.start);
        }
        self.tidy();
        Ok(d)
    }

    /// Move a cut's selected side to the playhead (Extend Edit).
    pub fn extend(&mut self, track: usize, at: i64, side: Side, to: i64, ripple: bool) -> Result<i64, TrimError> {
        self.trim(track, at, side, to - at, ripple)
    }

    /// Slip: the clip shows other source frames (`d` SOURCE frames later)
    /// in the same place. Clamped to the media. Returns frames slipped.
    pub fn slip(&mut self, track: usize, idx: usize, d: i64) -> i64 {
        let g = self.group(track, idx);
        let c = self.clip(g[0]).clone();
        let n = self.media[c.media].frames;
        let last = c.src_in + c.src_offset(c.len - 1, self.fps, self.media[c.media].fps);
        let (lo, hi) = (c.src_in.min(last), c.src_in.max(last));
        let d = d.clamp(-lo, n - 1 - hi);
        for (t, i) in g {
            self.tracks[t].clips[i].src_in += d;
        }
        self.tidy();
        d
    }

    /// Set a clip's speed. The source it covers stays (a faster clip gets
    /// shorter), its start stays, and it never grows into the next clip
    /// (it shows less source instead). Negative plays it backwards from its
    /// last frame. Returns the new length.
    pub fn set_speed(&mut self, track: usize, idx: usize, speed: f64) -> i64 {
        let conform = self.tracks[track].clips[idx].conform;
        self.retime(track, idx, speed, conform)
    }

    /// Conform a clip (and its linked clips) or undo it: one source frame
    /// per record frame, or real time again. As `set_speed`: the source it
    /// covers and its start stay, its length follows (a 30 fps clip in a
    /// 24 fps sequence gets 25% longer), never into the next clip. Returns
    /// the new length.
    pub fn set_conform(&mut self, track: usize, idx: usize, conform: bool) -> i64 {
        let speed = self.tracks[track].clips[idx].speed;
        self.retime(track, idx, speed, conform)
    }

    fn retime(&mut self, track: usize, idx: usize, speed: f64, conform: bool) -> i64 {
        if speed == 0.0 || !speed.is_finite() {
            return self.tracks[track].clips[idx].len;
        }
        let g = self.group(track, idx);
        let c = self.clip(g[0]).clone();
        let src_fps = self.media[c.media].fps;
        let span = (c.rate(self.fps, src_fps) * c.len as f64).abs();
        let ratio = Clip { conform, ..c.clone() }.ratio(self.fps, src_fps);
        let mut len = ((span / (speed.abs() * ratio)).round() as i64).max(1);
        for &(t, i) in &g {
            let next = self.tracks[t].clips.get(i + 1).map_or(i64::MAX, |n| n.start - c.start);
            len = len.min(next);
        }
        let flip = (speed < 0.0) != (c.speed < 0.0);
        let src_in = if flip { c.src_in + c.src_offset(c.len - 1, self.fps, src_fps) } else { c.src_in };
        for (t, i) in g {
            let x = &mut self.tracks[t].clips[i];
            x.speed = speed;
            x.conform = conform;
            x.len = len;
            x.src_in = src_in;
        }
        for t in &mut self.tracks {
            t.clips.sort_by_key(|c| c.start);
        }
        self.tidy();
        len
    }

    /// Set a clip's speed keeping record frame `anchor` still: the clip's
    /// length scales about it (its source stays, as `set_speed`), so a
    /// point inside the clip shows the same source before and after. Never
    /// overlaps a neighbour: the clip is cut to the room there is (showing
    /// less source). Returns (start, len).
    pub fn speed_about(&mut self, track: usize, idx: usize, speed: f64, anchor: i64) -> (i64, i64) {
        let c0 = self.tracks[track].clips[idx].clone();
        if speed == 0.0 || !speed.is_finite() {
            return (c0.start, c0.len);
        }
        let g = self.group(track, idx);
        let src_fps = self.media[c0.media].fps;
        let span = (c0.rate(self.fps, src_fps) * c0.len as f64).abs();
        let ratio = c0.ratio(self.fps, src_fps);
        let len = ((span / (speed.abs() * ratio)).round() as i64).max(1);
        let a = anchor.clamp(c0.start, c0.end());
        let mut start = a - ((a - c0.start) as f64 * len as f64 / c0.len as f64).round() as i64;
        let mut end = start + len;
        // The room between the neighbours (every track of the group).
        let (mut lo, mut hi) = (0, i64::MAX);
        for &(t, i) in &g {
            let tr = &self.tracks[t];
            if i > 0 {
                lo = lo.max(tr.clips[i - 1].end());
            }
            if let Some(n) = tr.clips.get(i + 1) {
                hi = hi.min(n.start);
            }
        }
        // What falls outside the room is cut off (the source it skips is
        // lost at the head, the tail simply ends).
        let skip = (lo - start).max(0);
        start = start.max(lo);
        end = end.min(hi).max(start + 1);
        let flip = (speed < 0.0) != (c0.speed < 0.0);
        let mut src_in = if flip { c0.src_in + c0.src_offset(c0.len - 1, self.fps, src_fps) } else { c0.src_in };
        let probe = Clip { speed, ..c0.clone() };
        src_in += probe.src_offset(skip, self.fps, src_fps);
        for (t, i) in g {
            let x = &mut self.tracks[t].clips[i];
            x.speed = speed;
            x.start = start;
            x.len = end - start;
            x.src_in = src_in;
        }
        for t in &mut self.tracks {
            t.clips.sort_by_key(|c| c.start);
        }
        self.tidy();
        (start, end - start)
    }

    /// Point a Wobble clip at version `v`: its render when there is one,
    /// else its source. The record frames it covers do not change; only
    /// what plays there (and `src_in` follows the media: render frames are
    /// counted from the exported range's first source frame).
    pub fn wobble_activate(&mut self, track: usize, idx: usize, v: u32) -> bool {
        let Some(w) = self.tracks[track].clips[idx].wobble.clone() else { return false };
        let c = &self.tracks[track].clips[idx];
        // Back to source frames from whatever plays now.
        let src = match w.renders.iter().find(|r| r.media == c.media) {
            Some(r) if c.media != w.source => c.src_in + r.first,
            _ => c.src_in,
        };
        let (media, src_in) = match w.render_of(v) {
            Some(r) => (r.media, src - r.first),
            None => (w.source, src),
        };
        let c = &mut self.tracks[track].clips[idx];
        c.media = media;
        c.src_in = src_in;
        if let Some(x) = c.wobble.as_mut() {
            x.active = v;
        }
        true
    }

    /// Set how a clip's picture sits in the raster (its whole group).
    pub fn set_reframe(&mut self, track: usize, idx: usize, r: crate::Reframe) {
        for (t, i) in self.group(track, idx) {
            self.tracks[t].clips[i].reframe = r;
        }
    }

    /// Put a dissolve of `len` frames on the cut at `at` (centred), clamped
    /// to the handles both clips have. Returns it, or why there is none.
    pub fn add_dissolve(&mut self, track: usize, at: i64, len: i64) -> Result<Dissolve, TrimError> {
        let (a, b) = self.tracks[track].cut(at);
        let (Some(a), Some(b)) = (a, b) else { return err("a dissolve needs a clip on both sides of the cut") };
        let (ca, cb) = (&self.tracks[track].clips[a], &self.tracks[track].clips[b]);
        let before = (len / 2).min(self.handles(cb).0).min(ca.len);
        let after = (len - len / 2).min(self.handles(ca).1).min(cb.len);
        if before + after == 0 {
            return err("no handles: neither clip has source beyond the cut");
        }
        let d = Dissolve { at, before, after };
        let t = &mut self.tracks[track];
        t.dissolves.retain(|x| x.at != at);
        t.dissolves.push(d);
        t.dissolves.sort_by_key(|x| x.at);
        Ok(d)
    }

    /// Set a dissolve's two sides (frames before and after its cut), each
    /// clamped to the handle there is. Returns it as set.
    pub fn set_dissolve_sides(&mut self, track: usize, at: i64, before: i64, after: i64) -> Result<Dissolve, TrimError> {
        let (a, b) = self.tracks[track].cut(at);
        let (Some(a), Some(b)) = (a, b) else { return err("a dissolve needs a clip on both sides of the cut") };
        let (ca, cb) = (&self.tracks[track].clips[a], &self.tracks[track].clips[b]);
        let before = before.clamp(0, self.handles(cb).0.min(ca.len));
        let after = after.clamp(0, self.handles(ca).1.min(cb.len));
        if before + after == 0 {
            return err("a dissolve needs at least one frame");
        }
        let d = Dissolve { at, before, after };
        let t = &mut self.tracks[track];
        t.dissolves.retain(|x| x.at != at);
        t.dissolves.push(d);
        t.dissolves.sort_by_key(|x| x.at);
        Ok(d)
    }

    /// Move clips (each given as (track, index); pass whole groups to keep
    /// picture and sound together) by `dt` frames and `dtrack` rows (+ =
    /// up: V1 toward V3). Video clips stay among V1-V3 and their sound on
    /// A1-A3 moves with them (V2 -> V3 takes A2 -> A3); sound on M stays.
    /// Sound moved on its own goes among A1..M (+ = toward A1, which is up
    /// on screen). The track move is clamped so everything lands on a track
    /// of its kind; nothing moves before frame 0. Overwrite: the clips
    /// replace what is under them where they land. `insert`: every track
    /// opens a gap there first (ripple on drop). Returns the (dt, dtrack)
    /// applied.
    pub fn move_clips(&mut self, picks: &[(usize, usize)], dt: i64, dtrack: i32, insert: bool) -> (i64, i32) {
        if picks.is_empty() {
            return (0, 0);
        }
        let mut moving: Vec<(usize, Clip)> = picks.iter().map(|&(t, i)| (t, self.tracks[t].clips[i].clone())).collect();
        let min_start = moving.iter().map(|(_, c)| c.start).min().unwrap_or(0);
        let dt = dt.max(-min_start);
        let with_video = moving.iter().any(|(t, _)| VIDEO_TRACKS.contains(t));
        let dtrack = clamp_track_move(&moving.iter().map(|(t, _)| *t).collect::<Vec<_>>(), dtrack, with_video);
        // Out of their old places (highest index first per track).
        let mut sorted = picks.to_vec();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.dedup();
        for (t, i) in sorted {
            self.tracks[t].clips.remove(i);
        }
        for (t, c) in &mut moving {
            c.start += dt;
            *t = moved_track(*t, dtrack, with_video);
        }
        if insert {
            let at = moving.iter().map(|(_, c)| c.start).min().unwrap_or(0);
            let end = moving.iter().map(|(_, c)| c.end()).max().unwrap_or(at);
            for t in 0..self.tracks.len() {
                self.split(t, at);
            }
            self.shift_from(at, end - at);
        }
        for (t, c) in moving {
            self.place(t, c);
        }
        self.tidy();
        (dt, dtrack)
    }

    /// Razor: cut the clip across `at` on `track` in two there, with its
    /// linked group (picture and sound cut together). True when something
    /// was cut.
    pub fn razor(&mut self, track: usize, at: i64) -> bool {
        let Some(i) = self.tracks[track].clips.iter().position(|c| c.start < at && at < c.end()) else { return false };
        let g = self.group(track, i);
        for (t, _) in g {
            self.split(t, at);
        }
        self.tidy();
        true
    }

    /// Razor on several tracks at once (each with its linked group): how
    /// many tracks had a clip across `at`.
    pub fn razor_tracks(&mut self, tracks: &[usize], at: i64) -> usize {
        let mut n = 0;
        for &t in tracks {
            let before = self.tracks.iter().map(|t| t.clips.len()).sum::<usize>();
            self.razor(t, at);
            if self.tracks.iter().map(|t| t.clips.len()).sum::<usize>() != before {
                n += 1;
            }
        }
        n
    }

    /// Slide: the clip (with its group) moves `d` frames along the
    /// timeline showing the same source, and the clips either side give
    /// and take: the one before ends later (or earlier), the one after
    /// starts later (or earlier), so the total length stays. Next to a gap
    /// the gap gives and takes instead. Clamped to the neighbours' lengths
    /// and handles. Returns the frames slid.
    pub fn slide(&mut self, track: usize, idx: usize, d: i64) -> i64 {
        let g = self.group(track, idx);
        let (mut lo, mut hi) = (i64::MIN, i64::MAX);
        for &(t, i) in &g {
            let tr = &self.tracks[t];
            let c = &tr.clips[i];
            lo = lo.max(-c.start);
            match tr.clips.get(i.wrapping_sub(1)).filter(|_| i > 0) {
                Some(p) if p.end() == c.start => {
                    lo = lo.max(-(p.len - 1));
                    hi = hi.min(self.handles(p).1);
                }
                Some(p) => lo = lo.max(p.end() - c.start),
                None => {}
            }
            match tr.clips.get(i + 1) {
                Some(n) if n.start == c.end() => {
                    hi = hi.min(n.len - 1);
                    lo = lo.max(-self.handles(n).0);
                }
                Some(n) => hi = hi.min(n.start - c.end()),
                None => {}
            }
        }
        let d = d.clamp(lo.min(0), hi.max(0));
        if d == 0 {
            return 0;
        }
        let (fps, media) = (self.fps, self.media.clone());
        for &(t, i) in &g {
            let tr = &mut self.tracks[t];
            let (s, e) = (tr.clips[i].start, tr.clips[i].end());
            if i > 0 && tr.clips[i - 1].end() == s {
                tr.clips[i - 1].len += d;
            }
            if let Some(n) = tr.clips.get_mut(i + 1).filter(|n| n.start == e) {
                n.src_in += n.src_offset(d, fps, media[n.media].fps);
                n.start += d;
                n.len -= d;
            }
            tr.clips[i].start += d;
            for x in &mut tr.dissolves {
                if x.at == s || x.at == e {
                    x.at += d;
                }
            }
        }
        self.tidy();
        d
    }

    /// Rate stretch: drag a clip's edge and its SPEED changes so the same
    /// source fills the new length (the other edge stays). `side` Out moves
    /// the end, In the start. Never overlaps a neighbour and never goes
    /// under a frame. Returns the new length.
    pub fn stretch(&mut self, track: usize, idx: usize, side: Side, d: i64) -> i64 {
        let g = self.group(track, idx);
        let c = self.clip(g[0]).clone();
        let mut len = match side {
            Side::In => c.len - d,
            _ => c.len + d,
        };
        for &(t, i) in &g {
            let tr = &self.tracks[t];
            match side {
                Side::In => {
                    let room = if i == 0 { c.start } else { c.start - tr.clips[i - 1].end() };
                    len = len.min(c.len + room);
                }
                _ => {
                    let room = tr.clips.get(i + 1).map_or(i64::MAX / 4, |n| n.start - c.end());
                    len = len.min(c.len + room);
                }
            }
        }
        let len = len.max(1);
        if len == c.len {
            return len;
        }
        let speed = c.speed * c.len as f64 / len as f64;
        let start = if side == Side::In { c.end() - len } else { c.start };
        for (t, i) in g {
            let x = &mut self.tracks[t].clips[i];
            x.speed = speed;
            x.len = len;
            x.start = start;
        }
        for t in &mut self.tracks {
            t.clips.sort_by_key(|c| c.start);
        }
        self.tidy();
        len
    }

    /// Duplicate clips (each (track, index), whole groups to keep picture
    /// and sound together) moved `dt` frames and `dtrack` rows, as
    /// `move_clips` places them; the originals stay. The copies get a new
    /// link. Returns the (dt, dtrack) applied.
    pub fn duplicate_clips(&mut self, picks: &[(usize, usize)], dt: i64, dtrack: i32, insert: bool) -> (i64, i32) {
        if picks.is_empty() {
            return (0, 0);
        }
        let mut moving: Vec<(usize, Clip)> = picks.iter().map(|&(t, i)| (t, self.tracks[t].clips[i].clone())).collect();
        let min_start = moving.iter().map(|(_, c)| c.start).min().unwrap_or(0);
        let dt = dt.max(-min_start);
        let with_video = moving.iter().any(|(t, _)| VIDEO_TRACKS.contains(t));
        let dtrack = clamp_track_move(&moving.iter().map(|(t, _)| *t).collect::<Vec<_>>(), dtrack, with_video);
        // One new link per old one.
        let mut links: Vec<(u64, u64)> = Vec::new();
        for (t, c) in &mut moving {
            let new = match links.iter().find(|l| l.0 == c.link) {
                Some(l) => l.1,
                None => {
                    let n = self.next_link;
                    self.next_link += 1;
                    links.push((c.link, n));
                    n
                }
            };
            c.link = new;
            c.start += dt;
            *t = moved_track(*t, dtrack, with_video);
        }
        if insert {
            let at = moving.iter().map(|(_, c)| c.start).min().unwrap_or(0);
            let end = moving.iter().map(|(_, c)| c.end()).max().unwrap_or(at);
            for t in 0..self.tracks.len() {
                self.split(t, at);
            }
            self.shift_from(at, end - at);
        }
        for (t, c) in moving {
            self.place(t, c);
        }
        self.tidy();
        (dt, dtrack)
    }

    pub fn remove_dissolve(&mut self, track: usize, at: i64) -> bool {
        let t = &mut self.tracks[track];
        let n = t.dissolves.len();
        t.dissolves.retain(|x| x.at != at);
        t.dissolves.len() != n
    }
}

/// Where track `t` goes when a move of `dtrack` rows is applied (see
/// `Sequence::move_clips`); `with_video`: the move carries a video clip.
pub fn moved_track(t: usize, dtrack: i32, with_video: bool) -> usize {
    if VIDEO_TRACKS.contains(&t) {
        (t as i32 + dtrack) as usize
    } else if with_video {
        if t == M {
            t
        } else {
            (t as i32 + dtrack) as usize
        }
    } else {
        (t as i32 - dtrack) as usize
    }
}

/// `dtrack` clamped so every track in `ts` lands on a track of its kind.
fn clamp_track_move(ts: &[usize], dtrack: i32, with_video: bool) -> i32 {
    let (mut lo, mut hi) = (i32::MIN, i32::MAX);
    for &t in ts {
        let (first, last) = if VIDEO_TRACKS.contains(&t) {
            (0, V3)
        } else if with_video {
            if t == M {
                continue;
            }
            (A1, A3)
        } else {
            // Sound alone: + moves toward A1 (down the index).
            let (a, b) = (AUDIO_TRACKS[0] as i32 - t as i32, M as i32 - t as i32);
            lo = lo.max(-b);
            hi = hi.min(-a);
            continue;
        };
        lo = lo.max(first as i32 - t as i32);
        hi = hi.min(last as i32 - t as i32);
    }
    if lo > hi {
        return 0;
    }
    dtrack.clamp(lo.min(0), hi.max(0))
}
