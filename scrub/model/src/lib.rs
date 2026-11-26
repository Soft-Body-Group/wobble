//! The edit: a sequence of tracks holding clips, and the edits that change
//! it. No UI, no decoding: the app draws it and plays it, this crate only
//! answers questions and applies edits, and everything here is tested.
//!
//! Time is in FRAMES at the sequence rate (`Fps`, an exact rational), so a
//! record position is an integer and an edit is frame accurate. A clip
//! names a media item, the source frame it starts on, where it sits in the
//! sequence, how long it lasts there, and its SPEED (1.0 normal, 2.0 twice
//! as fast, negative plays backwards; frames are sampled nearest-earlier, no
//! blending). A source at another rate is mapped by time.
//!
//! Tracks: V1-V3 (V3 on top: the topmost picture at a frame is what plays),
//! A1-A3 (A1 carries V1's sound, A2 V2's, A3 V3's) and M, a music track
//! nothing links to. Every audio track is mixed, with its own mute, solo
//! and gain. Every edit keeps clips on a track sorted and non-overlapping.
//! Clips made by one edit share a LINK (a video clip and its audio); the
//! clip edits (delete, trim, slip, speed) apply to the whole linked group,
//! so picture and sound never drift apart. Insert and ripple edits move
//! every track at once.
//!
//! Dissolves live on a track at a CUT (where one clip ends and the next
//! begins), `before` frames ahead of it and `after` past it. Through a
//! dissolve the outgoing clip plays on past its out and the incoming one
//! starts early, so both need HANDLES (source beyond what the cut shows);
//! a dissolve is clamped to the handles there are. A dissolve whose cut an
//! edit breaks is removed.

mod edit;
mod text;
mod three;
pub use edit::{moved_track, Side, TrimError};
pub use text::{format_srt, parse_srt, Align, ClipText, TextLayer, TextStyle, VAlign, PLACES};
pub use three::{resolve, Marks, Resolved};

use serde::{Deserialize, Deserializer, Serialize};
use std::path::PathBuf;

/// Frames per second, as an exact ratio.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fps {
    pub num: i64,
    pub den: i64,
}

impl Fps {
    pub const fn new(num: i64, den: i64) -> Self {
        Self { num, den }
    }

    pub fn as_f64(self) -> f64 {
        self.num as f64 / self.den.max(1) as f64
    }

    /// `n` frames at `self`, as frames at `to`, rounded down (exact when
    /// the rates match).
    /// The same rate (48/2 is 24/1).
    pub fn same(self, o: Fps) -> bool {
        self.num as i128 * o.den as i128 == o.num as i128 * self.den as i128
    }

    pub fn convert(self, n: i64, to: Fps) -> i64 {
        if self == to {
            return n;
        }
        let num = n as i128 * to.num as i128 * self.den as i128;
        let den = (to.den as i128 * self.num as i128).max(1);
        num.div_euclid(den) as i64
    }
}

/// A clip's media: what the app needs to open and draw it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Media {
    pub path: PathBuf,
    pub name: String,
    pub frames: i64,
    pub fps: Fps,
    pub has_audio: bool,
    /// Picture size (0 when unknown: media from before it was recorded).
    #[serde(default)]
    pub width: u32,
    #[serde(default)]
    pub height: u32,
    /// Made by Scrub, no file (the text clips' item).
    #[serde(default)]
    pub generated: bool,
    /// A sound file (WAV, MP3…): no picture, so an edit puts it on the
    /// targeted audio track only, never a video track.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub audio_only: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    Video,
    Audio,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Clip {
    /// Index into `Sequence::media`.
    pub media: usize,
    /// Source frame (at the media's rate) shown at `start`.
    pub src_in: i64,
    /// Record frame the clip starts on.
    pub start: i64,
    /// Record frames it lasts.
    pub len: i64,
    /// Clips made by one edit share a link (a video clip and its audio).
    pub link: u64,
    /// Playback speed: 1 normal, 2 double, -1 backwards.
    pub speed: f64,
    /// Conformed: one source frame per record frame (times `speed`)
    /// whatever the two rates, as a conform does. A 30 fps clip in a 24 fps
    /// sequence plays every frame, slower and longer, with no drops or
    /// repeats. Off: it plays at real time, frames dropped or repeated.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub conform: bool,
    /// How the picture sits in the sequence raster.
    #[serde(default)]
    pub reframe: Reframe,
    /// A text clip's words and style (`text`); `None` for footage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<ClipText>,
    /// A Wobble clip: its comp's versions (experimental; see
    /// docs/wobble-clips.md). `media` / `src_in` then name what PLAYS: the
    /// active version's render, or the source while there is none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wobble: Option<WobbleRef>,
    /// EXPERIMENT (exp/scrub-fold): the clip IS a wobble comp. The setup
    /// file (.wbl) whose viewed node the clip shows, live, in place of its
    /// footage; the comp reads the clip's source at the same frame numbers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wob: Option<String>,
}

/// A clip's Wobble comp: where its folder is, its versions, which is
/// active, and the source it was made from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WobbleRef {
    /// The folder name under the project's `.wobble` folder.
    pub uid: String,
    /// Every version made (1, 2, …) and which plays.
    pub versions: Vec<u32>,
    pub active: u32,
    /// The source media, and the source frames the comp renders: the used
    /// range plus handles (widened when a trim needs more).
    pub source: usize,
    pub first: i64,
    pub len: i64,
    /// Each rendered version: its media and the source frame its render
    /// starts on (render frame k is source frame `first + k`).
    #[serde(default)]
    pub renders: Vec<WobbleRender>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WobbleRender {
    pub version: u32,
    pub media: usize,
    pub first: i64,
}

impl WobbleRef {
    /// A version's render, if it has one.
    pub fn render_of(&self, v: u32) -> Option<WobbleRender> {
        self.renders.iter().find(|r| r.version == v).copied()
    }
}

/// How a clip's picture is placed in the sequence raster: fitted (all of it,
/// bars where the shapes differ) or filled (covers the raster, overflow
/// cropped), then zoomed and moved. `x`, `y` move the picture's centre by
/// that fraction of the raster's width / height. A 16:9 shot in a 9:16
/// sequence is typically filled and moved sideways onto its subject.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Reframe {
    pub fill: bool,
    pub zoom: f64,
    pub x: f64,
    pub y: f64,
}

impl Default for Reframe {
    fn default() -> Self {
        Reframe { fill: false, zoom: 1.0, x: 0.0, y: 0.0 }
    }
}

impl Reframe {
    pub fn is_default(&self) -> bool {
        *self == Reframe::default()
    }

    /// Where a `pw`x`ph` picture lands in a `rw`x`rh` raster, as
    /// (x, y, w, h) in raster pixels (it may run past the raster: the
    /// raster crops it).
    pub fn place(&self, (pw, ph): (u32, u32), (rw, rh): (u32, u32)) -> (f64, f64, f64, f64) {
        let (pw, ph, rw, rh) = (pw.max(1) as f64, ph.max(1) as f64, rw.max(1) as f64, rh.max(1) as f64);
        let base = if self.fill { (rw / pw).max(rh / ph) } else { (rw / pw).min(rh / ph) };
        let s = base * self.zoom.max(0.01);
        let (w, h) = (pw * s, ph * s);
        let (cx, cy) = (rw / 2.0 + self.x * rw, rh / 2.0 + self.y * rh);
        (cx - w / 2.0, cy - h / 2.0, w, h)
    }
}

impl Clip {
    pub fn end(&self) -> i64 {
        self.start + self.len
    }

    pub fn covers(&self, t: i64) -> bool {
        t >= self.start && t < self.end()
    }

    /// Source frames moved after `off` record frames (rounded down).
    pub fn src_offset(&self, off: i64, seq: Fps, src: Fps) -> i64 {
        if self.speed == 1.0 {
            return if self.conform { off } else { seq.convert(off, src) };
        }
        (off as f64 * self.speed * self.ratio(seq, src) + 1e-9).floor() as i64
    }

    /// Source frames per record frame (signed).
    pub fn rate(&self, seq: Fps, src: Fps) -> f64 {
        self.speed * self.ratio(seq, src)
    }

    /// Source frames per record frame at speed 1: the rates' ratio, or 1
    /// when conformed.
    pub fn ratio(&self, seq: Fps, src: Fps) -> f64 {
        if self.conform { 1.0 } else { src.as_f64() / seq.as_f64() }
    }

    /// The clip's speed against real time (what sound can follow): `speed`,
    /// times the conform's change when conformed.
    pub fn realtime_speed(&self, seq: Fps, src: Fps) -> f64 {
        if self.conform && !seq.same(src) { self.speed * seq.as_f64() / src.as_f64() } else { self.speed }
    }
}

/// A dissolve at a cut on one track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Dissolve {
    /// The cut: the outgoing clip ends here, the incoming one starts here.
    pub at: i64,
    pub before: i64,
    pub after: i64,
}

impl Dissolve {
    pub fn start(&self) -> i64 {
        self.at - self.before
    }
    pub fn end(&self) -> i64 {
        self.at + self.after
    }
    pub fn len(&self) -> i64 {
        self.before + self.after
    }
    pub fn covers(&self, t: i64) -> bool {
        t >= self.start() && t < self.end()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Track {
    /// Fixed by position (V1 V2 V3 A1 A2 A3 M); not stored, set again on load.
    #[serde(skip)]
    pub name: &'static str,
    pub kind: Kind,
    /// Sound: left out of the mix (audio tracks only).
    #[serde(default)]
    pub mute: bool,
    /// Sound: when any track is soloed only soloed tracks are heard.
    #[serde(default)]
    pub solo: bool,
    /// Sound: the track's gain in dB (0 = as recorded).
    #[serde(default)]
    pub gain_db: f32,
    /// Sorted by start, never overlapping.
    pub clips: Vec<Clip>,
    /// Sorted by cut.
    pub dissolves: Vec<Dissolve>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Sequence {
    pub fps: Fps,
    /// The record raster (width, height): the frame every picture is fitted
    /// into. Set from the first clip edited in unless chosen.
    #[serde(default)]
    pub raster: Option<(u32, u32)>,
    /// New edits of a clip at another rate are conformed (`Clip::conform`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub conform: bool,
    pub media: Vec<Media>,
    /// V1, V2, V3, A1, A2, A3, M.
    pub tracks: Vec<Track>,
    next_link: u64,
    /// Named text styles, shared by text clips (`text`).
    #[serde(default)]
    pub styles: Vec<TextStyle>,
    /// The project's colour: working space, view, per-file input spaces.
    #[serde(default)]
    pub colour: ColourSettings,
}

/// Colour management, wobble's model (`colour`): every picture is taken
/// from its file's INPUT space into the WORKING space (scene-linear) on
/// the GPU, and shown through the VIEW (a display transform, exposure,
/// gamma). Defaults keep what Scrub always showed: working Linear Rec.709,
/// view None (clamp) to an sRGB display, movies and stills as Rec.709 with
/// the sRGB curve, EXR as the working space, DPX as Cineon log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColourSettings {
    pub working: colour::color::Colorspace,
    pub view: colour::viewtf::ViewSettings,
    /// A file's input space when set by hand (else by its format).
    pub inputs: std::collections::BTreeMap<std::path::PathBuf, colour::color::InputSpace>,
}

impl Default for ColourSettings {
    fn default() -> Self {
        let mut view = colour::viewtf::ViewSettings { transform: colour::viewtf::ViewTransform::None, ..Default::default() };
        view.mirror_working(colour::color::Colorspace::LINEAR_REC709);
        ColourSettings { working: colour::color::Colorspace::LINEAR_REC709, view, inputs: Default::default() }
    }
}

impl ColourSettings {
    /// What a file of this kind holds when nobody said: by its format.
    pub fn default_for(path: &std::path::Path) -> Option<colour::color::Colorspace> {
        use colour::color::Colorspace;
        let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        match ext.as_str() {
            // Scene-linear: the working space itself.
            "exr" | "hdr" => None,
            // Log film scans (wobble's guess).
            "dpx" | "cin" => Some(Colorspace::CINEON),
            // Movies and display-referred stills: Rec.709 primaries with
            // the sRGB curve — what Scrub (and the Wobble clip round trip)
            // always showed them as, so a Rec.709 delivery looks as it did.
            _ => Some(Colorspace::SRGB),
        }
    }

    /// A file's input colourspace, resolved.
    pub fn input_of(&self, path: &std::path::Path) -> colour::color::Colorspace {
        use colour::color::InputSpace;
        match self.inputs.get(path).copied().unwrap_or(InputSpace::Default) {
            InputSpace::Working => self.working,
            InputSpace::Explicit(cs) => cs,
            InputSpace::Default => Self::default_for(path).unwrap_or(self.working),
        }
    }
}

pub const V1: usize = 0;
pub const V2: usize = 1;
pub const V3: usize = 2;
pub const A1: usize = 3;
pub const A2: usize = 4;
pub const A3: usize = 5;
/// The music track: sound no picture is linked to.
pub const M: usize = 6;
pub const VIDEO_TRACKS: [usize; 3] = [V1, V2, V3];
pub const AUDIO_TRACKS: [usize; 4] = [A1, A2, A3, M];
/// Every track, its name and kind, by position.
pub const TRACKS: [(&str, Kind); 7] =
    [("V1", Kind::Video), ("V2", Kind::Video), ("V3", Kind::Video), ("A1", Kind::Audio), ("A2", Kind::Audio), ("A3", Kind::Audio), ("M", Kind::Audio)];

/// The audio track that carries video track `v`'s sound (V1 -> A1 ...).
pub fn audio_of(v: usize) -> usize {
    v + A1
}

/// The video track an audio track is paired with (none for M).
pub fn video_of(a: usize) -> Option<usize> {
    (A1..=A3).contains(&a).then(|| a - A1)
}

/// A dB gain as a linear factor.
pub fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Which tracks an edit lands on: one video track (or none) and one audio
/// track (or none).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Targets {
    pub video: Option<usize>,
    /// Projects before version 2 stored `true` / `false` (A1 or none).
    #[serde(deserialize_with = "audio_target")]
    pub audio: Option<usize>,
}

fn audio_target<'de, D: Deserializer<'de>>(d: D) -> Result<Option<usize>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum T {
        Old(bool),
        New(Option<usize>),
    }
    Ok(match T::deserialize(d)? {
        T::Old(b) => b.then_some(A1),
        T::New(o) => o.filter(|a| AUDIO_TRACKS.contains(a)),
    })
}

impl Default for Targets {
    fn default() -> Self {
        Targets { video: Some(V1), audio: Some(A1) }
    }
}

impl Targets {
    /// Pick the video target. The sound follows it to its paired track
    /// (V2 -> A2) when it sat on a paired track; on M or off it stays.
    pub fn set_video(&mut self, v: Option<usize>) {
        self.video = v;
        if let (Some(v), Some(a)) = (v, self.audio) {
            if video_of(a).is_some() {
                self.audio = Some(audio_of(v));
            }
        }
    }

    /// Toggle a target: on when it was off (replacing the kind's other
    /// target), off when it was on.
    pub fn toggle(&mut self, track: usize) {
        if VIDEO_TRACKS.contains(&track) {
            let v = if self.video == Some(track) { None } else { Some(track) };
            self.set_video(v);
        } else if AUDIO_TRACKS.contains(&track) {
            self.audio = if self.audio == Some(track) { None } else { Some(track) };
        }
    }

    pub fn has(&self, track: usize) -> bool {
        self.video == Some(track) || self.audio == Some(track)
    }
}

/// What an edit puts in: a stretch of one media item.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Source {
    pub media: usize,
    pub src_in: i64,
    /// Record frames.
    pub len: i64,
    pub speed: f64,
    /// One source frame per record frame (`Clip::conform`).
    pub conform: bool,
}

impl Source {
    pub fn new(media: usize, src_in: i64, len: i64) -> Self {
        Self { media, src_in, len, speed: 1.0, conform: false }
    }
}

/// One decoded picture a frame needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layer {
    pub media: usize,
    pub frame: i64,
    pub reframe: Reframe,
}

/// What the record viewer shows at a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Picture {
    Black,
    One(Layer),
    /// A dissolve: `a` fading out, `b` in, `mix` of `b` (0..1).
    Mix(Layer, Layer, f32),
}

impl Track {
    fn new(name: &'static str, kind: Kind) -> Self {
        Self { name, kind, mute: false, solo: false, gain_db: 0.0, clips: Vec::new(), dissolves: Vec::new() }
    }

    pub fn at(&self, t: i64) -> Option<&Clip> {
        self.index_at(t).map(|i| &self.clips[i])
    }

    pub fn index_at(&self, t: i64) -> Option<usize> {
        let i = self.clips.partition_point(|c| c.end() <= t);
        (i < self.clips.len() && self.clips[i].covers(t)).then_some(i)
    }

    /// The clips meeting at a cut: (ending there, starting there).
    pub fn cut(&self, at: i64) -> (Option<usize>, Option<usize>) {
        (self.clips.iter().position(|c| c.end() == at), self.clips.iter().position(|c| c.start == at))
    }

    pub fn dissolve_at(&self, at: i64) -> Option<&Dissolve> {
        self.dissolves.iter().find(|d| d.at == at)
    }
}

impl Sequence {
    pub fn new(fps: Fps) -> Self {
        Self {
            fps,
            raster: None,
            conform: false,
            media: Vec::new(),
            tracks: TRACKS.iter().map(|&(n, k)| Track::new(n, k)).collect(),
            next_link: 1,
            styles: Vec::new(),
            colour: ColourSettings::default(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.iter().all(|t| t.clips.is_empty())
    }

    /// After loading: the track names (not stored) and a sane shape. An
    /// error says what is wrong with the document. A sequence from before
    /// A2, A3 and M (four tracks) gets them, empty.
    pub fn after_load(&mut self) -> Result<(), String> {
        if self.tracks.len() == 4 {
            for &(n, k) in &TRACKS[4..] {
                self.tracks.push(Track::new(n, k));
            }
        }
        if self.tracks.len() != TRACKS.len() {
            return Err(format!("expected {} tracks, found {}", TRACKS.len(), self.tracks.len()));
        }
        for (t, (name, kind)) in self.tracks.iter_mut().zip(TRACKS) {
            if t.kind != kind {
                return Err(format!("track {name} has the wrong kind"));
            }
            t.name = name;
        }
        if self.fps.num <= 0 || self.fps.den <= 0 {
            return Err("the frame rate is not positive".into());
        }
        let n = self.media.len();
        if self.tracks.iter().flat_map(|t| &t.clips).any(|c| c.media >= n) {
            return Err("a clip names media that is not in the list".into());
        }
        let ns = self.styles.len();
        if self.tracks.iter().flat_map(|t| &t.clips).any(|c| c.text.as_ref().is_some_and(|x| x.style >= ns)) {
            return Err("a text clip names a style that is not in the list".into());
        }
        let top = self.tracks.iter().flat_map(|t| &t.clips).map(|c| c.link).max().unwrap_or(0);
        self.next_link = self.next_link.max(top + 1);
        if !self.is_consistent() {
            return Err("clips overlap or a dissolve has no cut".into());
        }
        Ok(())
    }

    /// The media table index for `m`, adding it when new (by path).
    pub fn add_media(&mut self, m: Media) -> usize {
        if let Some(i) = self.media.iter().position(|x| x.path == m.path) {
            return i;
        }
        self.media.push(m);
        self.media.len() - 1
    }

    /// The audio tracks the mix hears and each one's linear gain: every
    /// unmuted track, or only the soloed ones when any is soloed.
    pub fn audible(&self) -> Vec<(usize, f32)> {
        let solo = AUDIO_TRACKS.iter().any(|&t| self.tracks[t].solo);
        AUDIO_TRACKS
            .iter()
            .filter(|&&t| if solo { self.tracks[t].solo } else { !self.tracks[t].mute })
            .map(|&t| (t, db_to_gain(self.tracks[t].gain_db)))
            .collect()
    }

    /// One past the last frame of anything.
    pub fn duration(&self) -> i64 {
        self.tracks.iter().filter_map(|t| t.clips.last()).map(Clip::end).max().unwrap_or(0)
    }

    fn src_fps(&self, c: &Clip) -> Fps {
        self.media[c.media].fps
    }

    /// The source frame clip `c` shows at record frame `t` (also outside
    /// the clip, for dissolves), clamped to the media.
    pub fn source_frame(&self, c: &Clip, t: i64) -> i64 {
        let f = c.src_in + c.src_offset(t - c.start, self.fps, self.src_fps(c));
        f.clamp(0, (self.media[c.media].frames - 1).max(0))
    }

    /// Record frames of source available BEFORE the clip's first frame and
    /// AFTER its last (what a trim or a dissolve can use).
    pub fn handles(&self, c: &Clip) -> (i64, i64) {
        if self.media[c.media].generated {
            // Generated (text): it can be as long as anyone likes.
            return (i64::MAX / 8, i64::MAX / 8);
        }
        let n = self.media[c.media].frames;
        let r = c.rate(self.fps, self.src_fps(c)).abs().max(1e-9);
        let last = c.src_in + c.src_offset(c.len - 1, self.fps, self.src_fps(c));
        let (before, after) = if c.speed >= 0.0 { (c.src_in, n - 1 - last) } else { (n - 1 - c.src_in, last) };
        ((before.max(0) as f64 / r + 1e-9).floor() as i64, (after.max(0) as f64 / r + 1e-9).floor() as i64)
    }

    /// The topmost video clip at `t` and the source frame it shows there.
    pub fn video_at(&self, t: i64) -> Option<(&Clip, i64)> {
        VIDEO_TRACKS.iter().rev().find_map(|&ti| self.tracks[ti].at(t).filter(|c| c.text.is_none())).map(|c| (c, self.source_frame(c, t)))
    }

    /// What plays at `t`: the topmost track with a picture there, and
    /// through a dissolve, both its clips.
    pub fn picture_at(&self, t: i64) -> Picture {
        for &ti in VIDEO_TRACKS.iter().rev() {
            let track = &self.tracks[ti];
            if let Some(d) = track.dissolves.iter().find(|d| d.covers(t)) {
                if let (Some(a), Some(b)) = track.cut(d.at) {
                    let (a, b) = (&track.clips[a], &track.clips[b]);
                    if a.text.is_some() || b.text.is_some() {
                        // A text side fades in `text_at`; the picture is
                        // whichever footage clip covers t, if any.
                        if let Some(c) = track.at(t).filter(|c| c.text.is_none()) {
                            return Picture::One(Layer { media: c.media, frame: self.source_frame(c, t), reframe: c.reframe });
                        }
                        continue;
                    }
                    let la = Layer { media: a.media, frame: self.source_frame(a, t), reframe: a.reframe };
                    let lb = Layer { media: b.media, frame: self.source_frame(b, t), reframe: b.reframe };
                    let mix = ((t - d.start()) as f32 + 0.5) / d.len() as f32;
                    return Picture::Mix(la, lb, mix);
                }
            }
            if let Some(c) = track.at(t).filter(|c| c.text.is_none()) {
                return Picture::One(Layer { media: c.media, frame: self.source_frame(c, t), reframe: c.reframe });
            }
        }
        Picture::Black
    }

    /// Every frame where a clip starts or ends, on any track, sorted.
    pub fn edit_points(&self) -> Vec<i64> {
        let mut v: Vec<i64> = self.tracks.iter().flat_map(|t| t.clips.iter().flat_map(|c| [c.start, c.end()])).collect();
        v.push(0);
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn next_edit(&self, t: i64) -> Option<i64> {
        self.edit_points().into_iter().find(|&p| p > t)
    }

    pub fn prev_edit(&self, t: i64) -> Option<i64> {
        self.edit_points().into_iter().rev().find(|&p| p < t)
    }

    /// The cut on `track` nearest `t`.
    pub fn nearest_cut(&self, track: usize, t: i64) -> Option<i64> {
        let tr = &self.tracks[track];
        tr.clips.iter().flat_map(|c| [c.start, c.end()]).min_by_key(|&p| (p - t).abs())
    }

    fn clips_for(&mut self, src: Source, at: i64, targets: Targets) -> Vec<(usize, Clip)> {
        let link = self.next_link;
        self.next_link += 1;
        let clip = Clip { media: src.media, src_in: src.src_in, start: at, len: src.len, link, speed: src.speed, conform: src.conform, reframe: Reframe::default(), text: None, wobble: None, wob: None };
        let mut v = Vec::new();
        // A sound file has no picture: its audio clip alone.
        if let Some(vt) = targets.video.filter(|_| !self.media[src.media].audio_only) {
            v.push((vt, clip.clone()));
        }
        if let Some(at) = targets.audio.filter(|_| self.media[src.media].has_audio) {
            v.push((at, clip));
        }
        v
    }

    /// Overwrite: the source replaces whatever the targeted tracks hold
    /// over its length; nothing moves.
    pub fn overwrite(&mut self, src: Source, at: i64, targets: Targets) {
        if src.len <= 0 {
            return;
        }
        let clips = self.clips_for(src, at, targets);
        for (ti, c) in clips {
            self.place(ti, c);
        }
        self.tidy();
    }

    /// Insert: every track opens a gap at `at` as long as the source
    /// (clips across `at` are cut), and the source goes into it on the
    /// targeted tracks.
    pub fn insert(&mut self, src: Source, at: i64, targets: Targets) {
        if src.len <= 0 {
            return;
        }
        for t in 0..self.tracks.len() {
            self.split(t, at);
        }
        self.shift_from(at, src.len);
        self.overwrite(src, at, targets);
    }

    /// Fit to fill: `src_len` SOURCE frames from `src_in` stretched (or
    /// squeezed) to exactly `rec_len` record frames, overwritten at `at`.
    pub fn overwrite_fit(&mut self, media: usize, src_in: i64, src_len: i64, at: i64, rec_len: i64, targets: Targets) {
        if let Some(src) = self.fit_source(media, src_in, src_len, rec_len) {
            self.overwrite(src, at, targets);
        }
    }

    /// Fit to fill as an insert: the same stretched source, everything
    /// from `at` on moved along to make room.
    pub fn insert_fit(&mut self, media: usize, src_in: i64, src_len: i64, at: i64, rec_len: i64, targets: Targets) {
        if let Some(src) = self.fit_source(media, src_in, src_len, rec_len) {
            self.insert(src, at, targets);
        }
    }

    /// `src_len` source frames from `src_in` played over `rec_len` record
    /// frames: the speed that takes.
    pub fn fit_source(&self, media: usize, src_in: i64, src_len: i64, rec_len: i64) -> Option<Source> {
        if rec_len <= 0 || src_len <= 0 {
            return None;
        }
        let conform = self.conform;
        let ratio = if conform { 1.0 } else { self.media[media].fps.as_f64() / self.fps.as_f64() };
        let speed = src_len as f64 / (rec_len as f64 * ratio);
        Some(Source { media, src_in, len: rec_len, speed, conform })
    }

    /// Lift: empty `[a, b)` on the targeted tracks, leaving a gap.
    pub fn lift(&mut self, a: i64, b: i64, targets: Targets) {
        let tracks: Vec<usize> = targets.video.into_iter().chain(targets.audio).collect();
        for t in tracks {
            self.clear(t, a, b);
        }
        self.tidy();
    }

    /// Every track's clips are sorted and do not overlap, and every
    /// dissolve sits on a real cut within its clips' handles.
    pub fn is_consistent(&self) -> bool {
        self.tracks.iter().all(|t| {
            t.clips.windows(2).all(|w| w[0].end() <= w[1].start)
                && t.clips.iter().all(|c| c.len > 0)
                && t.dissolves.iter().all(|d| {
                    let (a, b) = t.cut(d.at);
                    match (a, b) {
                        (Some(a), Some(b)) => {
                            let (a, b) = (&t.clips[a], &t.clips[b]);
                            d.after <= self.handles(a).1 && d.before <= self.handles(b).0 && d.len() > 0
                        }
                        _ => false,
                    }
                })
        })
    }

    // --- primitives the edits are built from ----------------------------------

    /// Make `[a, b)` empty on track `t`: clips inside go, clips across an
    /// end are trimmed, a clip across both is split in two.
    fn clear(&mut self, t: usize, a: i64, b: i64) {
        if a >= b {
            return;
        }
        let (fps, media) = (self.fps, &self.media);
        let clips = std::mem::take(&mut self.tracks[t].clips);
        let mut out = Vec::with_capacity(clips.len() + 1);
        for c in clips {
            if c.end() <= a || c.start >= b {
                out.push(c);
                continue;
            }
            if c.start < a {
                out.push(Clip { len: a - c.start, ..c.clone() });
            }
            if c.end() > b {
                let skip = b - c.start;
                let src_in = c.src_in + c.src_offset(skip, fps, media[c.media].fps);
                out.push(Clip { src_in, start: b, len: c.end() - b, ..c });
            }
        }
        self.tracks[t].clips = out;
    }

    /// Cut the clip across `at` on track `t` into two at `at`.
    fn split(&mut self, t: usize, at: i64) {
        let (fps, media) = (self.fps, &self.media);
        let tr = &mut self.tracks[t];
        if let Some(i) = tr.clips.iter().position(|c| c.start < at && at < c.end()) {
            let c = tr.clips[i].clone();
            let skip = at - c.start;
            tr.clips[i].len = skip;
            let src_in = c.src_in + c.src_offset(skip, fps, media[c.media].fps);
            tr.clips.insert(i + 1, Clip { src_in, start: at, len: c.len - skip, ..c });
        }
    }

    fn place(&mut self, t: usize, clip: Clip) {
        self.clear(t, clip.start, clip.end());
        let tr = &mut self.tracks[t];
        let i = tr.clips.partition_point(|c| c.start < clip.start);
        tr.clips.insert(i, clip);
    }

    /// Move every clip (and dissolve) starting at or after `at`, on every
    /// track, by `by`.
    fn shift_from(&mut self, at: i64, by: i64) {
        for tr in &mut self.tracks {
            for c in &mut tr.clips {
                if c.start >= at {
                    c.start += by;
                }
            }
            for d in &mut tr.dissolves {
                if d.at > at {
                    d.at += by;
                }
            }
        }
    }

    /// Drop dissolves whose cut is gone and clamp the rest to the handles.
    fn tidy(&mut self) {
        for ti in 0..self.tracks.len() {
            let ds = std::mem::take(&mut self.tracks[ti].dissolves);
            let mut keep = Vec::new();
            for mut d in ds {
                let (a, b) = self.tracks[ti].cut(d.at);
                if let (Some(a), Some(b)) = (a, b) {
                    let (ca, cb) = (&self.tracks[ti].clips[a], &self.tracks[ti].clips[b]);
                    d.after = d.after.min(self.handles(ca).1).min(cb.len);
                    d.before = d.before.min(self.handles(cb).0).min(ca.len);
                    if d.len() > 0 {
                        keep.push(d);
                    }
                }
            }
            keep.sort_by_key(|d| d.at);
            self.tracks[ti].dissolves = keep;
        }
    }
}

#[cfg(test)]
mod tests;
