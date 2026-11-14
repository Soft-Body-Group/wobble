//! Three-point editing: from whichever marks are set on the source and the
//! record side, work out what goes where.
//!
//! Each side has an in and an out (either may be unset) and a playhead.
//! The rules, as in every NLE:
//! - Missing source in: the source playhead. Missing record in: the record
//!   playhead.
//! - The LENGTH comes from the source when it has both marks, else from the
//!   record side when it has both, else from the source in to the end of the
//!   clip.
//! - Record out with no record in BACK-TIMES: the edit ends at the out.
//!   Source out with no source in does the same on the source.
//! - Four marks: the source decides the length (the record out is ignored).

use crate::Fps;

/// In and out marks; `out` is the LAST frame included (as an NLE shows it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Marks {
    pub inp: Option<i64>,
    pub out: Option<i64>,
}

impl Marks {
    pub fn clear(&mut self) {
        *self = Marks::default();
    }

    /// Set the in; an out before it goes.
    pub fn set_in(&mut self, f: i64) {
        self.inp = Some(f);
        if self.out.is_some_and(|o| o < f) {
            self.out = None;
        }
    }

    /// Set the out; an in after it goes.
    pub fn set_out(&mut self, f: i64) {
        self.out = Some(f);
        if self.inp.is_some_and(|i| i > f) {
            self.inp = None;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resolved {
    /// First source frame.
    pub src_in: i64,
    /// Record frame the edit starts on.
    pub rec_in: i64,
    /// Length in record frames.
    pub len: i64,
}

/// Work out an edit. Source frames are at `src_fps` (`src_len` of them),
/// record frames at `rec_fps`.
pub fn resolve(
    src: Marks,
    src_playhead: i64,
    src_len: i64,
    src_fps: Fps,
    rec: Marks,
    rec_playhead: i64,
    rec_fps: Fps,
) -> Option<Resolved> {
    let to_rec = |n: i64| src_fps.convert(n, rec_fps);
    let to_src = |n: i64| rec_fps.convert(n, src_fps);
    // Length in record frames, and which side fixes it.
    let (src_in, len) = match (src.inp, src.out) {
        (Some(i), Some(o)) => (i, to_rec(o - i + 1)),
        (None, Some(o)) => match (rec.inp, rec.out) {
            (Some(ri), Some(ro)) => {
                let len = ro - ri + 1;
                (o + 1 - to_src(len), len)
            }
            _ => {
                let i = src_playhead.min(o);
                (i, to_rec(o - i + 1))
            }
        },
        (i, None) => {
            let i = i.unwrap_or(src_playhead);
            match (rec.inp, rec.out) {
                (Some(ri), Some(ro)) => (i, ro - ri + 1),
                _ => (i, to_rec(src_len - i)),
            }
        }
    };
    let rec_in = match (rec.inp, rec.out) {
        (Some(ri), _) => ri,
        (None, Some(ro)) => ro + 1 - len,
        (None, None) => rec_playhead,
    };
    // Back-timed past the sequence start: the head is trimmed off, so the
    // out still lands where it was marked.
    let (src_in, rec_in, len) = if rec_in < 0 { (src_in + to_src(-rec_in), 0, len + rec_in) } else { (src_in, rec_in, len) };
    let src_in = src_in.max(0);
    // Never past the end of the source.
    let len = len.min(to_rec(src_len - src_in));
    (len > 0 && src_in < src_len).then_some(Resolved { src_in, rec_in, len })
}
