//! What a clip is (`SourceInfo`) and how frames come out of it (`Source`).
//!
//! Frames are indexed `0..frames`; `first` is the number a user sees for
//! index 0 (1001 for a sequence named shot.1001.exr, 0 for a movie).

use crate::frame::Frame;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct SourceInfo {
    pub path: PathBuf,
    /// Short label for the HUD ("prores 3840x2160", "exr sequence").
    pub label: String,
    pub width: u32,
    pub height: u32,
    pub frames: u64,
    /// The displayed number of index 0.
    pub first: i64,
    pub fps_num: i64,
    pub fps_den: i64,
    pub has_alpha: bool,
    pub has_audio: bool,
    /// Every frame decodes on its own (ProRes, DNxHR, image files). A
    /// long-GOP codec (H.264) is not: a frame needs the ones before it, so
    /// frames are decoded in runs, one decoder streaming forward.
    pub intra: bool,
}

impl SourceInfo {
    pub fn fps(&self) -> f64 {
        self.fps_num as f64 / self.fps_den.max(1) as f64
    }
}

/// Where a decode stream delivers. The source calls `wants` for every
/// picture it decodes on its way to a target (a long-GOP seek passes
/// through the frames before it), `passing` with the ones wanted, and
/// `done` with each target; `done` names the next target or stops.
pub trait Sink {
    fn wants(&mut self, frame: u64) -> bool;
    fn passing(&mut self, frame: u64, picture: Frame);
    fn done(&mut self, frame: u64, picture: Result<Frame>) -> Option<u64>;
}

pub trait Source: Send + Sync {
    fn info(&self) -> &SourceInfo;

    /// Decode one frame.
    fn decode(&self, frame: u64) -> Result<Frame>;

    /// The frames the decode prefetch wants, most urgent first; `playing`:
    /// a source with a read-ahead of its own runs it on past them (an
    /// image sequence's file reads). Default: nothing.
    fn prefetch(&self, _frames: &[u64], _playing: bool) {}

    /// The read-ahead's state, for a source that has one.
    fn net(&self) -> Option<crate::fetch::NetStats> {
        None
    }

    /// Decode `first`, then whatever `sink.done` names next, on one warm
    /// decoder. The default suits sources whose frames are independent.
    fn stream(&self, first: u64, sink: &mut dyn Sink) {
        let mut f = first;
        loop {
            let r = self.decode(f);
            match sink.done(f, r) {
                Some(n) => f = n,
                None => break,
            }
        }
    }
}

pub type SourceRef = Arc<dyn Source>;

/// Open a movie, or the image sequence a sample file belongs to.
pub fn open(path: &Path) -> Result<SourceRef> {
    if crate::sequence::is_image(path) {
        Ok(Arc::new(crate::sequence::Sequence::open(path)?))
    } else {
        Ok(Arc::new(crate::movie::Movie::open(path)?))
    }
}
