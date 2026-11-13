//! The EDIT: a timeline over shots, inside wobble itself.
//!
//! Scrub used to be a separate process talking to wobble over a socket. It
//! is now a *mode* of the one application: the same engine, the same node
//! editor, the same panels — with a timeline where wobble alone has a
//! per-shot playhead. Nothing here renders or talks to a peer; this module
//! owns the project (clips, versions, bin), the timeline's gestures, and a
//! handful of REQUESTS the host (`Gfx`) applies each frame.
//!
//! Shots are documents embedded in the project (`Project::shots`), one per
//! placed clip — a bare plate is a one-node shot — and the host loads the
//! shot under the playhead into its editor exactly as it loads a setup file.
//! Each shot's node ids live in their own range (`ID_STRIDE`), so footage
//! loaded for one shot is never mistaken for another's when the playhead
//! crosses a cut.

pub mod cache;
pub mod export;
pub mod ingest;
pub mod model;
pub mod thumbs;
pub mod timeline;

pub use model::{Clip, Media, Project};

use std::collections::HashMap;
use std::path::PathBuf;

/// Node ids of shot `k` are `k * ID_STRIDE + n`, n ≥ 1.
pub const ID_STRIDE: u64 = 1_000_000;

/// The SOURCE VIEWER's graph lives in a reserved shot range, so its plate
/// stays loaded like any other shot's and never collides with one.
pub const SOURCE_SHOT: u64 = 999_999;

/// How a source clip lands on the timeline.
#[derive(Clone, Copy, PartialEq)]
pub enum Insert {
    /// At the playhead (nudged past whatever blocks it).
    AtPlayhead,
    /// After the last clip.
    Append,
}

/// Which shot a node id belongs to.
pub fn shot_of_node(id: u64) -> u64 {
    id / ID_STRIDE
}

/// The plate `Read` a shot is minted with: always the first id in its range,
/// so a version bump knows which node to rewrite without searching.
pub fn plate_node(shot: u64) -> u64 {
    shot * ID_STRIDE + 1
}

/// Drag payload for a bin row → timeline drop.
pub struct BinDrag(pub u64);

pub enum TlDrag {
    Scrub,
    Trim {
        clip: usize,
        side: TrimSide,
        start_in: usize,
        start_out: usize,
        start_pos: usize,
        prev_end: usize,
        next_start: usize,
        start_x: f32,
        ppf: f32,
    },
    Roll { left: usize, start_out: usize, start_in: usize, b_start: usize, start_x: f32, ppf: f32 },
    Move { clip: usize, grab: i64 },
    Slip { clip: usize, start_in: usize, start_out: usize, start_x: f32, ppf: f32 },
    HBar,
}

#[derive(Clone, Copy, PartialEq)]
pub enum TrimSide {
    In,
    Out,
}

pub enum TlHit {
    Playhead,
    Ruler,
    HBar,
    Roll(usize),
    Trim(usize, TrimSide),
    Body(usize),
    Empty,
}

pub struct EditState {
    pub project: Project,
    pub project_path: Option<PathBuf>,
    /// The ProRes proxy tier's producer. Jobs are started on request, not
    /// on drop: live evaluation is the default and the proxy is a fallback.
    pub cache: cache::CacheManager,
    pub cache_keys: HashMap<u64, cache::CacheKey>,
    /// Timeline frame under the playhead.
    pub playhead: f64,
    pub selected: Option<usize>,
    /// Started on the first frame (it needs the egui context to request
    /// repaints when a sequence arrives).
    pub ingest_rx: Option<std::sync::mpsc::Receiver<PathBuf>>,
    pub export_status: export::ExportStatus,
    pub error: Option<String>,
    pub tl_drag: Option<TlDrag>,
    pub tl_rect: Option<egui::Rect>,
    pub tl_zoom: Option<f32>,
    pub tl_scroll: f32,

    // ---- requests to the host, consumed each frame ----------------------
    /// The UI moved the playhead; the host seeks (and swaps shots) to match.
    pub seek_request: Option<f64>,
    /// Clips placed since the host last looked: each needs a shot minted.
    pub new_clips: Vec<u64>,
    /// Clips whose plate binding changed (version bump, input colourspace):
    /// the host rewrites the shot's `Read`.
    pub rebind: Vec<u64>,
    /// Shots whose clips are gone: the host drops any parked context.
    pub removed_shots: Vec<u64>,
    /// Standalone scripts to import as a clip's shot: (clip uid, script).
    pub import_requests: Vec<(u64, PathBuf)>,
    /// "enter wobble": open the comp editor window on the current shot.
    pub enter_wobble: bool,
    /// Open the recursive footage browser (wobble's own) for the bin.
    pub browse_request: bool,
    /// Timeline + bin undo: whole-project snapshots, one per settled change
    /// (`track_history`). The shots' graphs keep wobble's own undo.
    pub history: Vec<Project>,
    pub future: Vec<Project>,
    /// The project as of the last recorded step, and its signature.
    last_snap: Option<(u64, Project)>,
    /// The signature last saved (or opened); differing = unsaved.
    saved_sig: Option<u64>,
    /// A shot's graph changed since the last save (the host sets it).
    pub shots_dirty: bool,
    /// sequence manipulator did not launch once: "add sequences…" goes
    /// straight to the browser for the rest of the session.
    pub seq_manip_missing: bool,
    /// Bin posters.
    pub thumbs: thumbs::Thumbs,
    /// Bigger posters in the bin.
    pub big_thumbs: bool,
    /// Name being typed for a new folder, when the field is open.
    pub new_folder: Option<String>,
    /// SOURCE VIEWER: the bin clip being viewed instead of the timeline.
    /// While set, the transport plays that clip and the timeline's
    /// playhead holds still.
    pub source: Option<u64>,
    /// Enter (Some(uid)) or leave (None) the source viewer.
    pub source_request: Option<Option<u64>>,
    /// Scrub within the source clip: a plate frame to seek to.
    pub source_seek: Option<u64>,
    /// Put the source clip (with its marks) on the timeline.
    pub insert_request: Option<Insert>,
    /// Which mark a source-strip drag took hold of.
    pub source_grab: Option<TrimSide>,
    /// Folders folded shut in the bin.
    pub collapsed: std::collections::HashSet<String>,
    /// Where each bin row's drag handle was drawn last frame (tests aim
    /// synthetic pointer events at it).
    pub handle_rects: HashMap<u64, egui::Rect>,
    /// Save / open asked for from the edit's own bar.
    pub save_request: Option<PathBuf>,
    pub open_request: Option<PathBuf>,

    // ---- transport bookkeeping ------------------------------------------
    /// Shot currently loaded in the editor (None = a gap, or nothing yet).
    pub current: Option<u64>,
    /// Clip (uid) the playhead is on.
    pub current_clip: Option<u64>,
    /// Last shot-local tick seen, to notice the player wrapping.
    pub last_local: Option<u64>,
    /// When the playhead sits in a gap: its [start, end) in timeline frames.
    pub gap: Option<(usize, usize)>,
}

impl EditState {
    /// An edit with no decoder behind it: the thumbnail worker still runs,
    /// it just never finds an image. For tests of the project model.
    #[cfg(test)]
    pub fn for_test() -> Self {
        struct NoDecoder;
        impl engine::decode::ExrDecoder for NoDecoder {
            fn read_header(&self, _: &std::path::Path) -> anyhow::Result<engine::decode::ImageInfo> {
                anyhow::bail!("no decoder in tests")
            }
            fn decode_into(
                &self,
                _: &std::path::Path,
                _: engine::decode::ImageInfo,
                _: usize,
                _: &mut [half::f16],
            ) -> anyhow::Result<()> {
                anyhow::bail!("no decoder in tests")
            }
        }
        Self::new(std::sync::Arc::new(NoDecoder))
    }

    pub fn new(decoder: std::sync::Arc<dyn engine::decode::ExrDecoder>) -> Self {
        Self {
            project: Project::default(),
            project_path: None,
            cache: cache::CacheManager::new(),
            cache_keys: HashMap::new(),
            playhead: 0.0,
            selected: None,
            ingest_rx: None,
            export_status: export::ExportStatus::default(),
            error: None,
            tl_drag: None,
            tl_rect: None,
            tl_zoom: None,
            tl_scroll: 0.0,
            seek_request: None,
            new_clips: Vec::new(),
            rebind: Vec::new(),
            removed_shots: Vec::new(),
            import_requests: Vec::new(),
            enter_wobble: false,
            browse_request: false,
            history: Vec::new(),
            future: Vec::new(),
            last_snap: None,
            saved_sig: None,
            shots_dirty: false,
            seq_manip_missing: false,
            thumbs: thumbs::Thumbs::new(decoder),
            big_thumbs: false,
            new_folder: None,
            source: None,
            source_request: None,
            source_seek: None,
            insert_request: None,
            source_grab: None,
            collapsed: Default::default(),
            handle_rects: HashMap::new(),
            save_request: None,
            open_request: None,
            current: None,
            current_clip: None,
            last_local: None,
            gap: None,
        }
    }

    /// Drain sequences sent from sequence manipulator (starting the listener
    /// on first call).
    pub fn poll_ingest(&mut self, ctx: &egui::Context) {
        let rx = self.ingest_rx.get_or_insert_with(|| ingest::start(ctx.clone()));
        let mut got = Vec::new();
        while let Ok(p) = rx.try_recv() {
            got.push(p);
        }
        for p in got {
            self.add_clip_from_sample(p);
        }
    }

    /// The frame a clip's poster is decoded from: the middle of its media.
    pub fn poster_path(clip: &Clip) -> PathBuf {
        match clip.media() {
            Media::Frames(frames) => frames[frames.len() / 2].clone(),
            Media::Movie { path, .. } => path.clone(),
            Media::Blank { tag } => tag.clone(),
        }
    }

    /// Open sequence manipulator: it does the recursive search, and sends
    /// what you pick back over `ingest.sock` (see `ingest.rs`). It is the
    /// bin's loader — a separate app on purpose, since finding footage is
    /// its whole job.
    /// Record a history step when the project has changed since the last
    /// one. Call once per frame with `settled` false while a gesture is in
    /// flight, so a drag is ONE step (its end state), not one per frame.
    pub fn track_history(&mut self, settled: bool) {
        if !settled {
            return;
        }
        let sig = self.project.signature();
        match &self.last_snap {
            Some((s, _)) if *s == sig => {}
            Some(_) => {
                let (_, prev) = self.last_snap.replace((sig, self.project.clone())).unwrap();
                self.history.push(prev);
                if self.history.len() > 100 {
                    self.history.remove(0);
                }
                self.future.clear();
            }
            None => self.last_snap = Some((sig, self.project.clone())),
        }
        if self.saved_sig.is_none() {
            self.saved_sig = Some(sig);
        }
    }

    /// A fresh history whose saved state is now: after an open, or entering
    /// with an empty edit.
    pub fn reset_history(&mut self) {
        let sig = self.project.signature();
        self.history.clear();
        self.future.clear();
        self.last_snap = Some((sig, self.project.clone()));
        self.saved_sig = Some(sig);
        self.shots_dirty = false;
    }

    pub fn mark_saved(&mut self) {
        self.saved_sig = Some(self.project.signature());
        self.shots_dirty = false;
    }

    /// Unsaved work: the edit differs from what was saved or opened, or a
    /// shot's graph was edited since.
    pub fn is_dirty(&self) -> bool {
        self.shots_dirty || self.saved_sig.is_some_and(|s| s != self.project.signature())
    }

    pub fn undo(&mut self) -> bool {
        let Some(prev) = self.history.pop() else { return false };
        self.future.push(self.project.clone());
        self.restore(prev);
        true
    }

    pub fn redo(&mut self) -> bool {
        let Some(next) = self.future.pop() else { return false };
        self.history.push(self.project.clone());
        self.restore(next);
        true
    }

    /// Put the timeline and bin back as they were. The shots' graphs are
    /// NOT rolled back — they have their own undo, and the edits made in
    /// them since must survive — but a shot a removal took away comes back
    /// from the snapshot, and one nothing plays any more is dropped.
    fn restore(&mut self, snap: Project) {
        let shots = std::mem::take(&mut self.project.shots);
        self.project = Project { shots, ..snap.clone() };
        for (id, doc) in snap.shots {
            self.project.shots.entry(id).or_insert(doc);
        }
        let used: std::collections::HashSet<u64> =
            self.project.clips.iter().chain(self.project.bin.iter()).filter_map(|c| c.shot).collect();
        let orphans: Vec<u64> =
            self.project.shots.keys().copied().filter(|id| !used.contains(id) && Some(*id) != self.current).collect();
        for id in orphans {
            self.project.shots.remove(&id);
            self.removed_shots.push(id);
        }
        if self.source.is_some_and(|uid| !self.project.bin.iter().any(|c| c.uid == uid)) {
            self.source_request = Some(None);
        }
        self.last_snap = Some((self.project.signature(), self.project.clone()));
        self.selected = None;
        self.tl_drag = None;
        self.clamp_playhead();
    }

    /// "add sequences…": sequence manipulator when it is installed (its
    /// recursive search sends picks back over the ingest socket), else
    /// wobble's own footage browser, which lands picks in the bin too. Not
    /// having it is not an error: the browser is the loader everyone has.
    pub fn launch_seq_manip(&mut self) {
        if !self.seq_manip_missing && launch_seq_manip_app() {
            return;
        }
        self.seq_manip_missing = true;
        self.browse_request = true;
    }

    /// Request posters for every bin clip that lacks one.
    pub fn want_posters(&mut self) {
        for c in self.project.bin.iter().filter(|c| !c.is_blank()) {
            self.thumbs.request(c.uid, Self::poster_path(c));
        }
    }

    /// The path a shot's plate `Read` is minted with: a `#`-padded pattern
    /// for a frame sequence (so the reader resolves exactly these frames,
    /// not everything numbered in the directory), the file itself for a
    /// movie.
    pub fn plate_path(clip: &Clip) -> String {
        match clip.media() {
            Media::Frames(frames) => frames
                .first()
                .map(|f| pattern_of(f))
                .unwrap_or_default(),
            Media::Movie { path, .. } => path.to_string_lossy().into_owned(),
            Media::Blank { .. } => String::new(),
        }
    }

    /// A blank wobble clip in the bin: no footage, its shot an empty
    /// graph (minted when it is first placed, like any clip's).
    pub fn add_blank_clip(&mut self) {
        let n = self.project.bin.iter().filter(|c| c.is_blank()).count() + 1;
        self.project.bin.push(Clip::blank(&format!("blank {n}"), model::BLANK_DEFAULT_LEN));
    }

    pub fn add_clip_from_sample(&mut self, sample: PathBuf) {
        let sample = sample.canonicalize().unwrap_or(sample);
        if self.project.bin.iter().any(|c| c.versions.iter().any(|v| v.media.contains(&sample))) {
            log::info!("bin already holds {}", sample.display());
            return;
        }
        match Clip::from_sample(&sample) {
            Ok(clip) => self.project.bin.push(clip),
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    /// Place a bin clip on the timeline at (or nearest to) `desired`,
    /// carrying the marks set in the clip player. The instance INHERITS the
    /// bin clip's shot; if the bin clip has none yet, the host mints one for
    /// it (`new_clips` holds BIN uids) and every instance picks it up.
    /// Returns the instance's uid.
    pub fn place_from_bin(&mut self, uid: u64, desired: i64) -> Option<u64> {
        let clip = self.project.bin.iter().find(|c| c.uid == uid)?;
        let needs_shot = clip.shot.is_none();
        let mut inst = clip.instantiate();
        inst.start = model::resolve_start(&self.project.clips, inst.len(), desired, None);
        let inst_uid = inst.uid;
        self.project.clips.push(inst);
        self.selected = self.project.sort_timeline(inst_uid);
        if needs_shot && !self.new_clips.contains(&uid) {
            self.new_clips.push(uid);
        }
        Some(inst_uid)
    }

    pub fn append_from_bin(&mut self, uid: u64) {
        self.place_from_bin(uid, self.project.total_len() as i64);
    }

    /// The shot every instance of a bin clip shares, once minted.
    pub fn shot_of_master(&self, master: u64) -> Option<u64> {
        self.project.bin.iter().find(|c| c.uid == master).and_then(|c| c.shot)
    }

    /// Remove a timeline clip. Its SHOT stays: the bin clip owns that, and
    /// other instances (or a later placement) still need it.
    pub fn remove_clip(&mut self, idx: usize) {
        if idx >= self.project.clips.len() {
            return;
        }
        let c = self.project.clips.remove(idx);
        self.cache_keys.remove(&c.uid);
        self.selected = None;
        self.clamp_playhead();
    }

    /// Remove a bin entry, every timeline instance of it, and its shot —
    /// the bin clip is the owner, so this is where the graph goes.
    pub fn remove_bin(&mut self, uid: u64) {
        if let Some(entry) = self.project.bin.iter().find(|c| c.uid == uid) {
            let ident = entry.identity().to_path_buf();
            let shot = entry.shot;
            let doomed: Vec<usize> = self
                .project
                .clips
                .iter()
                .enumerate()
                .filter(|(_, c)| c.master == Some(uid) || c.identity() == ident)
                .map(|(i, _)| i)
                .collect();
            for i in doomed.into_iter().rev() {
                self.remove_clip(i);
            }
            if let Some(s) = shot {
                self.project.shots.remove(&s);
                self.removed_shots.push(s);
            }
        }
        self.project.bin.retain(|c| c.uid != uid);
        self.cache_keys.remove(&uid);
        self.selected = None;
        self.clamp_playhead();
    }

    fn clamp_playhead(&mut self) {
        let total = self.project.total_len();
        self.playhead = self.playhead.clamp(0.0, total.saturating_sub(1).max(0) as f64);
        self.seek_request = Some(self.playhead);
    }

    pub fn zoom_anchored(&mut self, new_ppf: f32, view_w: f32) {
        let new_ppf = new_ppf.clamp(0.02, 200.0);
        let Some(old) = self.tl_zoom else {
            self.tl_zoom = Some(new_ppf);
            return;
        };
        let vis_old = view_w / old;
        let ph = self.playhead as f32;
        let anchor = if ph >= self.tl_scroll && ph <= self.tl_scroll + vis_old {
            ph
        } else {
            self.tl_scroll + vis_old * 0.5
        };
        let frac = ((anchor - self.tl_scroll) / vis_old.max(1e-6)).clamp(0.0, 1.0);
        self.tl_scroll = anchor - frac * (view_w / new_ppf);
        self.tl_zoom = Some(new_ppf);
    }

    pub fn cache_state(&self, uid: u64) -> Option<cache::CacheState> {
        let k = *self.cache_keys.get(&uid)?;
        self.cache.state(k)
    }

    /// The clip under the playhead and the plate frame it shows.
    #[allow(dead_code)]
    pub fn at_playhead(&self) -> Option<(usize, usize)> {
        let total = self.project.total_len();
        if total == 0 {
            return None;
        }
        let global = (self.playhead.floor() as usize).min(total - 1);
        self.project.locate(global).map(|(ci, local)| (ci, self.project.clips[ci].in_off + local))
    }

    /// Move the playhead from the UI: records the request for the host.
    pub fn set_playhead(&mut self, f: f64) {
        self.playhead = f.max(0.0);
        self.seek_request = Some(self.playhead);
    }
}

/// `…/frame.0042.exr` → `…/frame.####.exr`: the last run of digits before
/// the extension becomes the frame field, padded to its width.
pub fn pattern_of(sample: &std::path::Path) -> String {
    let s = sample.to_string_lossy().into_owned();
    let (stem, ext) = match s.rfind('.') {
        Some(i) if !s[i..].contains('/') => (&s[..i], &s[i..]),
        _ => (s.as_str(), ""),
    };
    let bytes = stem.as_bytes();
    let mut end = bytes.len();
    while end > 0 && !bytes[end - 1].is_ascii_digit() {
        end -= 1;
    }
    if end == 0 {
        return s;
    }
    let mut start = end;
    while start > 0 && bytes[start - 1].is_ascii_digit() {
        start -= 1;
    }
    format!("{}{}{}{}", &stem[..start], "#".repeat(end - start), &stem[end..], ext)
}

/// Launch sequence manipulator; false when it is not installed (or this is
/// not a Mac, where it is the only build). `open -a` asks LaunchServices,
/// so a bundle anywhere on disk is found, and a missing one fails without
/// side effects. `$WOBBLE_SEQ_MANIP` names a dev bundle; a path baked in
/// here would ship inside every build.
fn launch_seq_manip_app() -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    let dev = std::env::var("WOBBLE_SEQ_MANIP").ok().filter(|p| std::path::Path::new(p).exists());
    let target = dev.as_deref().unwrap_or("seq_manip");
    std::process::Command::new("open")
        .args(["-a", target])
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_replaces_the_frame_digits_only() {
        let p = std::path::Path::new("/x/shot_v002/frame.0042.exr");
        assert_eq!(pattern_of(p), "/x/shot_v002/frame.####.exr", "not the version digits");
        let p = std::path::Path::new("/x/plate_1001.dpx");
        assert_eq!(pattern_of(p), "/x/plate_####.dpx");
        let p = std::path::Path::new("/x/clip.mov");
        assert_eq!(pattern_of(p), "/x/clip.mov", "no frame field: unchanged");
    }

    #[test]
    fn node_ids_are_namespaced_per_shot() {
        assert_eq!(shot_of_node(plate_node(7)), 7);
        assert_eq!(shot_of_node(plate_node(7) + 999), 7);
        assert_ne!(shot_of_node(plate_node(7)), shot_of_node(plate_node(8)));
    }
}

#[cfg(test)]
mod place_tests {
    use super::*;
    use crate::edit::model::Clip;

    fn clip(name: &str, frames: usize) -> Clip {
        Clip::for_test(name, frames)
    }

    /// Marks set in the clip player are what lands on the timeline.
    #[test]
    fn placing_respects_the_marked_range() {
        let mut e = EditState::for_test();
        e.project.bin.push(clip("shotA", 100));
        let uid = e.project.bin[0].uid;
        e.project.bin[0].in_off = 20;
        e.project.bin[0].out_off = 39;

        let inst = e.place_from_bin(uid, 0).expect("placed");
        let c = e.project.clips.iter().find(|c| c.uid == inst).unwrap();
        assert_eq!((c.in_off, c.out_off), (20, 39), "the marks came with it");
        assert_eq!(c.len(), 20, "20 frames on the timeline, not the whole 100");
        assert_eq!(c.end(), 20);
    }

    /// One bin clip, two placements, ONE graph: cutting a clip up leaves both
    /// halves driven by the same wobble script. A duplicate in the bin is how
    /// you get a second.
    #[test]
    fn instances_of_one_bin_clip_share_its_shot() {
        let mut e = EditState::for_test();
        e.project.bin.push(clip("shotA", 50));
        let uid = e.project.bin[0].uid;

        let a = e.place_from_bin(uid, 0).unwrap();
        assert_eq!(e.new_clips, vec![uid], "the BIN clip is what needs a shot");
        // The host mints and assigns to the master.
        e.project.bin[0].shot = Some(7);
        for c in e.project.clips.iter_mut() {
            c.shot = Some(7);
        }
        e.new_clips.clear();

        let b = e.place_from_bin(uid, 100).unwrap();
        assert!(e.new_clips.is_empty(), "the shot already exists — nothing to mint");
        let shots: Vec<Option<u64>> =
            [a, b].iter().map(|id| e.project.clips.iter().find(|c| c.uid == *id).unwrap().shot).collect();
        assert_eq!(shots, vec![Some(7), Some(7)], "both instances, one graph");

        // Removing one instance must NOT take the graph with it.
        let idx = e.project.clips.iter().position(|c| c.uid == a).unwrap();
        e.remove_clip(idx);
        assert_eq!(e.project.bin[0].shot, Some(7), "the bin clip still owns it");
        assert!(e.removed_shots.is_empty(), "no shot was dropped");

        // A duplicate is a fresh master with no shot.
        let dup = e.project.bin[0].duplicate_master();
        assert_eq!(dup.shot, None, "duplicating in the bin is how you get a second graph");
        assert_ne!(dup.uid, uid);
    }
}

