//! Text clips: titles and subtitles.
//!
//! A text clip is an ordinary clip on a video track whose media is the
//! sequence's generated "text" item (no file) and which carries its WORDS
//! (`ClipText::text`) and the index of a named STYLE (`TextStyle`): font,
//! size, colour, alignment, line spacing, outline, shadow, box and the
//! position in the frame. Content and style are separate on purpose: many
//! clips share one style, so retyping a clip never moves it, and editing
//! the style restyles (and repositions) every clip using it — duplicate a
//! subtitle (alt-drag), retype it, and it sits exactly where the last one
//! did.
//!
//! Text clips are transparent where they draw nothing: `picture_at` looks
//! through them to the picture below, and `text_at` lists the text to draw
//! over it, bottom to top, each with its opacity (a dissolve on a text
//! clip's cut fades it).

use crate::{Clip, Media, Reframe, Sequence, VIDEO_TRACKS};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Align {
    Left,
    #[default]
    Center,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VAlign {
    Top,
    #[default]
    Middle,
    Bottom,
}

/// How text looks and where it sits. Sizes are fractions of the raster's
/// HEIGHT and the position a fraction of its width / height, so a style
/// means the same thing at any resolution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextStyle {
    pub name: String,
    /// A TTF / OTF / TTC file; `None`: the built-in face.
    pub font: Option<PathBuf>,
    /// The face within a collection (TTC).
    pub font_index: u32,
    /// Cap-to-descender size as a fraction of the raster height.
    pub size: f32,
    /// RGBA, straight alpha.
    pub color: [u8; 4],
    pub align: Align,
    /// Lines' spacing as a multiple of the font's own.
    pub line_spacing: f32,
    /// Outline width as a fraction of `size` (0: none).
    pub outline: f32,
    pub outline_color: [u8; 4],
    /// A soft drop shadow down and right.
    pub shadow: bool,
    pub shadow_color: [u8; 4],
    /// A box behind the block.
    pub boxed: bool,
    pub box_color: [u8; 4],
    /// The box's padding as a fraction of `size`.
    pub box_pad: f32,
    /// Where the block's anchor sits (fractions of the raster): the anchor
    /// is its left / centre / right edge by `align`, and its top / middle /
    /// bottom by `valign`.
    pub x: f32,
    pub y: f32,
    pub valign: VAlign,
}

impl Default for TextStyle {
    fn default() -> Self {
        TextStyle {
            name: "title".into(),
            font: None,
            font_index: 0,
            size: 0.08,
            color: [255, 255, 255, 255],
            align: Align::Center,
            line_spacing: 1.0,
            outline: 0.0,
            outline_color: [0, 0, 0, 255],
            shadow: true,
            shadow_color: [0, 0, 0, 160],
            boxed: false,
            box_color: [0, 0, 0, 150],
            box_pad: 0.3,
            x: 0.5,
            y: 0.5,
            valign: VAlign::Middle,
        }
    }
}

/// The places a style can be put with one click (title-safe margins: 10%
/// of the frame; action-safe for subtitles: 5%).
pub const PLACES: [(&str, f32, f32, Align, VAlign); 5] = [
    ("centred", 0.5, 0.5, Align::Center, VAlign::Middle),
    ("title top", 0.5, 0.1, Align::Center, VAlign::Top),
    ("lower third", 0.1, 0.72, Align::Left, VAlign::Top),
    ("subtitle", 0.5, 0.92, Align::Center, VAlign::Bottom),
    ("top left", 0.1, 0.1, Align::Left, VAlign::Top),
];

impl TextStyle {
    pub fn subtitle() -> Self {
        TextStyle { name: "subtitle".into(), size: 0.055, outline: 0.0, shadow: false, boxed: true, x: 0.5, y: 0.92, valign: VAlign::Bottom, ..Default::default() }
    }

    /// Put it at one of `PLACES`.
    pub fn place(&mut self, i: usize) {
        if let Some(&(_, x, y, a, v)) = PLACES.get(i) {
            (self.x, self.y, self.align, self.valign) = (x, y, a, v);
        }
    }
}

/// A text clip's words and style.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClipText {
    pub text: String,
    /// Index into `Sequence::styles`.
    pub style: usize,
    /// Per-letter (word, line) animators, bottom to top.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub anims: Vec<crate::TextAnim>,
}

/// One text to draw over the picture at a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct TextLayer {
    pub text: String,
    pub style: TextStyle,
    pub opacity: f32,
    /// The clip's animators, and the clip frame (0..len) they are seen at.
    pub anims: Vec<crate::TextAnim>,
    pub frame: i64,
    pub len: i64,
}

impl TextLayer {
    /// Whether its animators are moving now (else it draws as its settled,
    /// cacheable self).
    pub fn moving(&self) -> bool {
        crate::textanim::moving(&self.anims, self.frame, self.len)
    }
}

impl Sequence {
    /// The generated media item text clips point at (added on first use).
    pub fn text_media(&mut self) -> usize {
        if let Some(i) = self.media.iter().position(|m| m.generated) {
            return i;
        }
        self.media.push(Media { path: PathBuf::new(), name: "text".into(), frames: i64::MAX / 4, fps: self.fps, has_audio: false, width: 0, height: 0, generated: true, audio_only: false });
        self.media.len() - 1
    }

    /// The styles, with the two defaults made on first use.
    pub fn ensure_styles(&mut self) {
        if self.styles.is_empty() {
            self.styles = vec![TextStyle::default(), TextStyle::subtitle()];
        }
    }

    /// A style by name.
    pub fn style_named(&self, name: &str) -> Option<usize> {
        self.styles.iter().position(|s| s.name.eq_ignore_ascii_case(name))
    }

    /// A text clip of `len` frames on video track `track` at `at`
    /// (overwriting what is there), in style `style`. Returns its index.
    pub fn add_text(&mut self, track: usize, at: i64, len: i64, text: &str, style: usize) -> Option<usize> {
        if !VIDEO_TRACKS.contains(&track) || len <= 0 {
            return None;
        }
        self.ensure_styles();
        let style = style.min(self.styles.len() - 1);
        let media = self.text_media();
        let link = self.next_link;
        self.next_link += 1;
        let clip = Clip { media, src_in: 0, start: at.max(0), len, link, speed: 1.0, conform: false, reframe: Reframe::default(), grade: None, text: Some(ClipText { text: text.to_string(), style, anims: Vec::new() }), wobble: None, wob: None };
        self.place(track, clip);
        self.tidy();
        self.tracks[track].clips.iter().position(|c| c.start == at.max(0))
    }

    /// The text to draw at `t`, bottom to top: text clips on video tracks
    /// ABOVE the topmost picture there (all of them over black), each with
    /// its opacity. A dissolve on a cut fades a text clip out (outgoing)
    /// or in (incoming) across it.
    pub fn text_at(&self, t: i64) -> Vec<TextLayer> {
        let floor = VIDEO_TRACKS.iter().rev().find(|&&ti| self.tracks[ti].at(t).is_some_and(|c| c.text.is_none())).copied();
        let mut out = Vec::new();
        for &ti in VIDEO_TRACKS.iter() {
            if floor.is_some_and(|f| ti < f) {
                continue;
            }
            let tr = &self.tracks[ti];
            // A clip covering t, or one a dissolve reaches into t from beside it.
            let mut push = |c: &Clip, opacity: f32| {
                if let (Some(ct), true) = (&c.text, opacity > 0.0) {
                    if let Some(s) = self.styles.get(ct.style) {
                        let frame = (t - c.start).clamp(0, (c.len - 1).max(0));
                        out.push(TextLayer { text: ct.text.clone(), style: s.clone(), opacity: opacity.min(1.0), anims: ct.anims.clone(), frame, len: c.len });
                    }
                }
            };
            let d = tr.dissolves.iter().find(|d| d.covers(t));
            match d.map(|d| (d, tr.cut(d.at))) {
                Some((d, (Some(a), Some(b)))) => {
                    let mix = ((t - d.start()) as f32 + 0.5) / d.len() as f32;
                    push(&tr.clips[a], 1.0 - mix);
                    push(&tr.clips[b], mix);
                }
                _ => {
                    if let Some(c) = tr.at(t) {
                        push(c, 1.0);
                    }
                }
            }
        }
        out
    }

    /// Replace a text clip's animators (nothing on a clip that is not
    /// text).
    pub fn set_text_anims(&mut self, track: usize, idx: usize, anims: Vec<crate::TextAnim>) {
        if let Some(t) = self.tracks.get_mut(track).and_then(|tr| tr.clips.get_mut(idx)).and_then(|c| c.text.as_mut()) {
            t.anims = anims;
        }
    }

    /// Text clips on `track` as (start, end, text): what an SRT export writes.
    pub fn texts_on(&self, track: usize) -> Vec<(i64, i64, String)> {
        self.tracks[track].clips.iter().filter_map(|c| c.text.as_ref().map(|t| (c.start, c.end(), t.text.clone()))).collect()
    }
}

/// SubRip: (start seconds, end seconds, text) per cue. Lenient: CRLF,
/// a BOM, missing indices, commas or dots before the milliseconds.
pub fn parse_srt(s: &str) -> Vec<(f64, f64, String)> {
    let s = s.trim_start_matches('\u{feff}').replace("\r\n", "\n").replace('\r', "\n");
    let mut out = Vec::new();
    for block in s.split("\n\n") {
        let mut lines = block.lines().filter(|l| !l.trim().is_empty()).peekable();
        if lines.peek().is_some_and(|l| l.trim().chars().all(|c| c.is_ascii_digit())) {
            lines.next();
        }
        let Some(times) = lines.next() else { continue };
        let Some((a, b)) = times.split_once("-->") else { continue };
        let (Some(a), Some(b)) = (srt_time(a), srt_time(b.split_whitespace().next().unwrap_or(""))) else { continue };
        let text: Vec<&str> = lines.collect();
        if b > a {
            out.push((a, b, text.join("\n")));
        }
    }
    out
}

fn srt_time(s: &str) -> Option<f64> {
    let s = s.trim().replace(',', ".");
    let mut parts = s.split(':');
    let (h, m, sec) = (parts.next()?.parse::<f64>().ok()?, parts.next()?.parse::<f64>().ok()?, parts.next()?.parse::<f64>().ok()?);
    Some(h * 3600.0 + m * 60.0 + sec)
}

/// Cues (start seconds, end seconds, text) as SubRip.
pub fn format_srt(cues: &[(f64, f64, String)]) -> String {
    let ts = |t: f64| {
        let ms = (t.max(0.0) * 1000.0).round() as u64;
        format!("{:02}:{:02}:{:02},{:03}", ms / 3_600_000, (ms / 60_000) % 60, (ms / 1000) % 60, ms % 1000)
    };
    cues.iter().enumerate().map(|(i, (a, b, t))| format!("{}\n{} --> {}\n{}\n", i + 1, ts(*a), ts(*b), t)).collect::<Vec<_>>().join("\n")
}
