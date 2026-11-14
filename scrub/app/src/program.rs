//! Playing the sequence: the record viewer's frames come from the clips'
//! own frame servers, one per media file, through the same decode, cache
//! and prefetch path as the source viewer.
//!
//! Each redraw the prefetch window is laid over the SEQUENCE (the playhead
//! and the frames it will show next, in the play direction), each of its
//! frames is resolved to (clip media, source frame), and every server is
//! handed exactly its share, most urgent first. A cut a second ahead is
//! therefore already decoding while the frames before it play: playback
//! does not stall at a cut. A server with no share gets an empty list,
//! which cancels its queued work.

use media::server::{wanted_list, PlayState};
use media::{Cache, FrameRef, FrameServer};
use model::{Layer, Picture, Reframe, Sequence};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// How far `pick` walks from the playhead toward the picture.
const PICK_SPAN: i64 = 64;
/// Prefetch depth over the sequence while playing.
const AHEAD: u64 = 48;

pub enum Pick {
    /// Nothing at this frame (a gap, or past the end).
    Black,
    /// A frame to show; `exact` when it is the playhead's. Through a
    /// dissolve, `mix` is the incoming frame and its weight.
    /// `media`: the clip's media item (its file's colourspace).
    Frame { key: u64, at: i64, frame: FrameRef, exact: bool, reframe: Reframe, media: usize, mix: Option<(u64, FrameRef, f32, Reframe, usize)> },
    /// Keep what is on screen (the frame is still decoding).
    Keep,
}

/// A frame's pictures, when every one of them is decoded.
enum Ready {
    Black,
    One(u64, FrameRef, Reframe, usize),
    Mix((u64, FrameRef, Reframe, usize), (u64, FrameRef, Reframe, usize), f32),
}

pub struct Program {
    cache: Arc<Cache>,
    /// By the media's ORIGINAL path; a server may read its optimised copy.
    servers: HashMap<PathBuf, FrameServer>,
}

impl Program {
    pub fn new(cache: Arc<Cache>) -> Self {
        Self { cache, servers: HashMap::new() }
    }

    /// Open a server for every media item clips use, drop the rest.
    pub fn sync(&mut self, seq: &Sequence, settings: &media::optimise::Settings) {
        // Generated media (text) has no file to decode.
        let used: Vec<&PathBuf> = seq.tracks.iter().flat_map(|t| t.clips.iter().filter(|c| !seq.media[c.media].generated).map(|c| &seq.media[c.media].path)).collect();
        self.servers.retain(|p, _| used.contains(&p));
        for p in used {
            if !self.servers.contains_key(p) {
                let read = media::optimise::find(p, settings).unwrap_or_else(|| p.clone());
                match media::open(&read) {
                    Ok(src) => {
                        self.servers.insert(p.clone(), FrameServer::new(src, self.cache.clone()));
                    }
                    Err(e) => log::warn!("cannot open {}: {e:#}", p.display()),
                }
            }
        }
    }

    /// Forget every server (a purge or an optimise changed what to read).
    pub fn reset(&mut self) {
        self.servers.clear();
    }

    fn server(&self, seq: &Sequence, media: usize) -> Option<&FrameServer> {
        self.servers.get(&seq.media[media].path)
    }

    /// The file a media item is actually read from (its optimised copy
    /// when one matched at open).
    pub fn read_path(&self, original: &std::path::Path) -> Option<&std::path::Path> {
        self.servers.get(original).map(|s| s.source().info().path.as_path())
    }

    /// One picture, if decoded (the trim monitor).
    pub fn layer_frame(&self, seq: &Sequence, l: Layer) -> Option<(u64, FrameRef)> {
        self.layer(seq, l)
    }

    fn layer(&self, seq: &Sequence, l: Layer) -> Option<(u64, FrameRef)> {
        let s = self.server(seq, l.media)?;
        s.get(l.frame as u64).map(|f| (key(s, l.frame as u64), f))
    }

    /// Frame `t`'s pictures, if all are decoded.
    fn ready(&self, seq: &Sequence, t: i64) -> Option<Ready> {
        match seq.picture_at(t) {
            Picture::Black => Some(Ready::Black),
            // EXPERIMENT (fold): a wob clip's picture is the comp's, live
            // (the record viewer shows it); its footage is not decoded.
            Picture::One(_) if seq.video_at(t).is_some_and(|(c, _)| c.wob.is_some()) => Some(Ready::Black),
            Picture::One(l) => self.layer(seq, l).map(|(k, f)| Ready::One(k, f, l.reframe, l.media)),
            Picture::Mix(a, b, m) => {
                let (ka, fa) = self.layer(seq, a)?;
                let (kb, fb) = self.layer(seq, b)?;
                Some(Ready::Mix((ka, fa, a.reframe, a.media), (kb, fb, b.reframe, b.media), m))
            }
        }
    }

    /// Hand every server its share of the prefetch window: each frame's
    /// pictures (both sides of a dissolve), at the frames a clip's speed
    /// actually samples.
    /// `extra`: pictures needed first whatever the playhead (the trim
    /// monitor's two frames).
    pub fn feed(&self, seq: &Sequence, st: PlayState, extra: &[Layer]) {
        let len = seq.duration().max(1) as u64;
        let wanted = wanted_list(PlayState { frame: st.frame.min(len - 1), ..st }, len, AHEAD);
        let mut lists: HashMap<&PathBuf, Vec<u64>> = HashMap::new();
        let mut gate: Option<(&PathBuf, u64)> = None;
        for l in extra {
            lists.entry(&seq.media[l.media].path).or_default().push(l.frame as u64);
        }
        for (i, &t) in wanted.iter().enumerate() {
            let layers = match seq.picture_at(t as i64) {
                Picture::Black => vec![],
                Picture::One(_) if seq.video_at(t as i64).is_some_and(|(c, _)| c.wob.is_some()) => vec![],
                Picture::One(l) => vec![l],
                Picture::Mix(a, b, _) => vec![a, b],
            };
            for l in layers {
                let p = &seq.media[l.media].path;
                let v = lists.entry(p).or_default();
                if !v.contains(&(l.frame as u64)) {
                    v.push(l.frame as u64);
                }
                if i == 0 && !st.playing && gate.is_none() {
                    gate = Some((p, l.frame as u64));
                }
            }
        }
        for (p, s) in &self.servers {
            let list = lists.get(p).map(Vec::as_slice).unwrap_or(&[]);
            let g = gate.filter(|(gp, _)| *gp == p).map(|(_, f)| f);
            s.set_list(list, g, 0..0);
        }
    }

    /// What the record viewer shows for playhead `t`, the picture being
    /// `shown`: the playhead's frame, else the decoded frame nearest it
    /// between the picture and the playhead, else keep the picture.
    pub fn pick(&self, seq: &Sequence, t: i64, shown: Option<i64>) -> Pick {
        let as_pick = |r: Ready, at: i64, exact: bool| match r {
            Ready::Black => Pick::Black,
            Ready::One(key, frame, reframe, media) => Pick::Frame { key, at, frame, exact, reframe, media, mix: None },
            Ready::Mix((ka, fa, ra, ma), (kb, fb, rb, mb), m) => Pick::Frame { key: ka, at, frame: fa, exact, reframe: ra, media: ma, mix: Some((kb, fb, m, rb, mb)) },
        };
        if let Some(r) = self.ready(seq, t) {
            return as_pick(r, t, true);
        }
        let Some(shown) = shown else { return Pick::Keep };
        let step = if shown < t { -1 } else { 1 };
        let mut u = t;
        for _ in 0..PICK_SPAN {
            u += step;
            if u == shown {
                break;
            }
            if let Some(r) = self.ready(seq, u) {
                if !matches!(r, Ready::Black) {
                    return as_pick(r, u, false);
                }
            }
        }
        Pick::Keep
    }

    /// Is frame `t` ready to show (a gap or the end always is)?
    pub fn is_cached(&self, seq: &Sequence, t: i64) -> bool {
        self.ready(seq, t).is_some()
    }

    /// Block until the playhead's pictures are in (headless checks).
    pub fn wait_for(&self, seq: &Sequence, t: i64, timeout: std::time::Duration) {
        let layers = match seq.picture_at(t) {
            Picture::Black => vec![],
            Picture::One(l) => vec![l],
            Picture::Mix(a, b, _) => vec![a, b],
        };
        for l in layers {
            if let Some(s) = self.server(seq, l.media) {
                s.wait_for(l.frame as u64, timeout);
            }
        }
    }

    /// Frames decoded so far and in the cache, summed over the servers.
    pub fn stats(&self) -> (u64, usize, f32) {
        let mut decoded = 0;
        let mut busy = 0;
        let mut ms = 0.0f32;
        for s in self.servers.values() {
            let st = s.stats();
            decoded += st.decoded;
            busy += st.queued + st.in_flight;
            ms = ms.max(st.decode_ms);
        }
        (decoded, busy, ms)
    }

    /// Image sequences' read-ahead, summed over the clips' files (only
    /// sequences on a share, or under the simulator, count).
    pub fn net(&self) -> Option<media::fetch::NetStats> {
        let mut out: Option<media::fetch::NetStats> = None;
        for s in self.servers.values() {
            let Some(n) = s.source().net().filter(|n| n.network) else { continue };
            let o = out.get_or_insert(media::fetch::NetStats { network: true, ..Default::default() });
            o.bytes_per_sec += n.bytes_per_sec;
            o.ahead += n.ahead;
            o.frame_bytes = o.frame_bytes.max(n.frame_bytes);
            o.local_hits += n.local_hits;
            o.remote_reads += n.remote_reads;
            o.queued += n.queued;
        }
        out
    }
}

/// A viewer slot's key for a frame: which file, which frame.
pub fn key(s: &FrameServer, f: u64) -> u64 {
    s.id().rotate_left(24) ^ f
}
