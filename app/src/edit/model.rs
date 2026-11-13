//! Timeline data model: an edit is an ordered list of clips, each pointing at
//! one VERSION of a piece of media — a file sequence or a movie (ProRes,
//! H.264, …). Versions are discovered from disk with a "v###" token in the
//! filename or the parent directory name, so version management is
//! stateless: a rescan is the database.
//!
//! Colour: every clip carries an INPUT colourspace (Wobble's own presets).
//! `None` means auto — the same per-extension guess Wobble's Reads show.
//! Decode paths transform input -> scene-linear ACEScg (the working space).

use anyhow::{anyhow, Context as _, Result};
use engine::color::{Colorspace, InputSpace};
use engine::decode::{AnyDecoder, ExrDecoder as _};
use serde::{Deserialize, Serialize};

/// The per-format default for a clip of this extension; `None` = working.
pub fn default_input(ext: &str) -> Option<Colorspace> {
    let f = engine::color::format_key(ext)?;
    match crate::colorui::defaults().input_for_format(f) {
        InputSpace::Explicit(cs) => Some(cs),
        _ => None,
    }
}
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub enum Media {
    Frames(Vec<PathBuf>),
    Movie { path: PathBuf, len: usize },
    /// No footage: a clip whose shot is a wobble graph from scratch. `tag`
    /// stands in for a path wherever one is the clip's identity (dedupe,
    /// bin removal, the project file); it names no file.
    Blank { tag: PathBuf },
}

/// How far a blank clip's out point may be dragged: it has no media to run
/// out of, so its "length" is headroom and the trim is the real length.
pub const BLANK_HEADROOM: usize = 100_000;
/// A new blank clip's length, in frames.
pub const BLANK_DEFAULT_LEN: usize = 100;

impl Media {
    pub fn len(&self) -> usize {
        match self {
            Media::Frames(f) => f.len(),
            Media::Movie { len, .. } => *len,
            Media::Blank { .. } => BLANK_HEADROOM,
        }
    }

    /// Representative path: identity for dedupe/cascade, the sample saved in
    /// project files, and the anchor for version rescans.
    pub fn sample(&self) -> &Path {
        match self {
            Media::Frames(f) => &f[0],
            Media::Movie { path, .. } => path,
            Media::Blank { tag } => tag,
        }
    }

    pub fn contains(&self, p: &Path) -> bool {
        match self {
            Media::Frames(f) => f.iter().any(|x| x == p),
            Media::Movie { path, .. } => path == p,
            Media::Blank { tag } => tag == p,
        }
    }

    pub fn extension(&self) -> String {
        self.sample()
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default()
    }
}

#[derive(Clone)]
pub struct Version {
    pub label: String, // "v012", or "—" for unversioned
    pub number: i64,
    pub media: Media,
}

/// Session-unique identity for one clip INSTANCE (a bin entry, or one use on
/// the timeline). Adding a bin clip to the timeline clones it under a new
/// uid, so each placement trims independently.
fn next_uid() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone)]
pub struct Clip {
    pub uid: u64,
    pub name: String,
    pub versions: Vec<Version>, // sorted ascending by number
    pub current: usize,
    pub in_off: usize,  // inclusive offset into the version's media
    pub out_off: usize, // inclusive
    /// Input colourspace; None = auto (per-extension guess, like Wobble).
    pub input: Option<Colorspace>,
    /// Timeline position (frame where this placement begins). Clips are kept
    /// sorted by it and never overlap; uncovered ranges are GAPS that play
    /// black. Meaningless for bin entries.
    pub start: usize,
    /// The shot this clip plays: an id into the project's embedded shot
    /// documents. Every placed clip has one — a bare plate is a one-node
    /// shot — so the timeline is uniformly a list of subgraphs.
    pub shot: Option<u64>,
    /// A project written by the standalone app pointed at a shot SCRIPT on
    /// disk; the app imports it as the embedded shot on open, then clears
    /// this. Never persisted again.
    pub legacy_script: Option<PathBuf>,
    /// The BIN clip this timeline instance came from. The bin clip owns the
    /// shot, so every instance of it is driven by the same wobble graph;
    /// duplicating it in the bin is how you get a second graph.
    pub master: Option<u64>,
    /// Which bin folder this clip sits in ("" = the root). A name, not a
    /// path: one level of folders is what an edit's bin needs.
    pub folder: String,
}

impl Clip {
    /// A copy of this clip as a fresh instance: own uid, own placement, the
    /// bin clip's trim (the marks set in the clip player) and — deliberately
    /// — the bin clip's SHOT, so two instances of one bin clip are driven by
    /// one wobble graph.
    pub fn instantiate(&self) -> Self {
        let mut c = self.clone();
        c.uid = next_uid();
        c.master = Some(self.uid);
        c
    }

    /// A bare clip over a made-up frame list, for tests.
    #[cfg(test)]
    pub fn for_test(name: &str, frames: usize) -> Self {
        let media = Media::Frames(
            (0..frames).map(|i| PathBuf::from(format!("/x/{name}.{i:04}.exr"))).collect(),
        );
        Self {
            uid: next_uid(),
            name: name.into(),
            versions: vec![Version { label: "v001".into(), number: 1, media }],
            current: 0,
            in_off: 0,
            out_off: frames.saturating_sub(1),
            input: None,
            start: 0,
            shot: None,
            legacy_script: None,
            master: None,
            folder: String::new(),
        }
    }

    /// A separate bin entry on the same media: its own uid and NO shot, so
    /// placing it mints a second graph.
    pub fn duplicate_master(&self) -> Self {
        let mut c = self.clone();
        c.uid = next_uid();
        c.master = None;
        c.shot = None;
        c.name = format!("{} copy", self.name);
        if self.is_blank() {
            // Nothing on disk to share: its own identity, or removing one
            // would take the other's clips with it.
            let len = c.out_off;
            c.versions = Clip::blank("", 1).versions;
            c.out_off = len;
        }
        c
    }

    pub fn from_sample(sample: &Path) -> Result<Self> {
        let is_movie = sample
            .extension()
            .is_some_and(|e| {
                let e = e.to_string_lossy().to_ascii_lowercase();
                engine::decode::MOVIE_EXTS.contains(&e.as_str())
            });
        let (name, versions) = if is_movie {
            scan_movie_versions(sample)?
        } else {
            scan_seq_versions(sample)?
        };
        // Start on the version that contains the sample file.
        let current = versions
            .iter()
            .position(|v| v.media.contains(sample))
            .unwrap_or(versions.len() - 1);
        let n = versions[current].media.len();
        Ok(Self {
            uid: next_uid(),
            name,
            versions,
            current,
            in_off: 0,
            out_off: n.saturating_sub(1),
            input: None,
            start: 0,
            shot: None,
            legacy_script: None,
            master: None,
            folder: String::new(),
        })
    }

    /// A clip with no footage, `frames` long: its shot starts as an empty
    /// wobble graph. The tag is unique across sessions (it is saved), so two
    /// blank clips never share an identity.
    pub fn blank(name: &str, frames: usize) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let mut c = Self::blank_tagged(PathBuf::from(format!("blank:{nanos:x}-{}", next_uid())));
        c.name = name.into();
        c.out_off = frames.clamp(1, BLANK_HEADROOM) - 1;
        c
    }

    fn blank_tagged(tag: PathBuf) -> Self {
        Self {
            uid: next_uid(),
            name: "blank".into(),
            versions: vec![Version { label: "—".into(), number: 0, media: Media::Blank { tag } }],
            current: 0,
            in_off: 0,
            out_off: BLANK_DEFAULT_LEN - 1,
            input: None,
            start: 0,
            shot: None,
            legacy_script: None,
            master: None,
            folder: String::new(),
        }
    }

    pub fn is_blank(&self) -> bool {
        matches!(self.media(), Media::Blank { .. })
    }

    /// Repoint this placement at different media (the clip's Wobble comp
    /// OUTPUT, once rendered), keeping identity-of-use: uid, position, trim
    /// (clamped), input pick and the comp link survive.
    #[allow(dead_code)]
    pub fn switch_media(&mut self, sample: &Path) -> Result<()> {
        let fresh = Clip::from_sample(sample)?;
        self.versions = fresh.versions;
        self.current = fresh.current;
        let n = self.media().len().max(1);
        self.in_off = self.in_off.min(n - 1);
        self.out_off = self.out_off.min(n - 1).max(self.in_off);
        Ok(())
    }

    /// One past the last timeline frame this placement covers.
    pub fn end(&self) -> usize {
        self.start + self.len()
    }

    pub fn version(&self) -> &Version {
        &self.versions[self.current]
    }

    pub fn media(&self) -> &Media {
        &self.version().media
    }

    /// What this clip is, independent of version and trim: the first
    /// version's sample. Removing a bin entry removes timeline clips that
    /// share this.
    pub fn identity(&self) -> &Path {
        self.versions[0].media.sample()
    }

    /// The colourspace decoded pixels are IN: the explicit pick, else the
    /// per-format default (`colour…`), else the working space (identity).
    pub fn effective_input(&self) -> Colorspace {
        self.input.or_else(|| default_input(&self.media().extension())).unwrap_or(Colorspace::ACESCG)
    }

    pub fn len(&self) -> usize {
        self.out_off - self.in_off + 1
    }

    pub fn latest(&self) -> usize {
        let mut best = 0;
        for (i, v) in self.versions.iter().enumerate() {
            if v.number >= self.versions[best].number {
                best = i;
            }
        }
        best
    }

    pub fn set_version(&mut self, idx: usize) {
        self.current = idx.min(self.versions.len() - 1);
        let n = self.media().len().max(1);
        self.in_off = self.in_off.min(n - 1);
        self.out_off = self.out_off.min(n - 1).max(self.in_off);
    }

    /// Re-discover versions on disk, keeping the current selection if it
    /// still exists.
    pub fn rescan(&mut self) {
        let sample = self.media().sample().to_path_buf();
        if let Ok(clip) = Clip::from_sample(&sample) {
            let label = self.version().label.clone();
            self.versions = clip.versions;
            self.current = self
                .versions
                .iter()
                .position(|v| v.label == label)
                .unwrap_or_else(|| self.latest());
            self.set_version(self.current);
        }
    }
}

#[derive(Clone)]
pub struct Project {
    pub fps: f32,
    /// The edit: ordered, butt-joined.
    pub clips: Vec<Clip>,
    /// The media pool. Ingest lands here; the timeline is built from it.
    pub bin: Vec<Clip>,
    /// The shots this edit owns, as setup documents (see `ProjectFile`).
    /// Opaque at this level: the app parses one into its graph when the
    /// playhead reaches it and writes it back when it leaves.
    pub shots: std::collections::BTreeMap<u64, serde_json::Value>,
    /// Bin folders, in display order — kept even when empty.
    pub folders: Vec<String>,
}

impl Default for Project {
    fn default() -> Self {
        Self { fps: 24.0, clips: Vec::new(), bin: Vec::new(), shots: Default::default(), folders: Vec::new() }
    }
}

impl Project {
    /// Timeline length: the end of the last clip (gaps included).
    pub fn total_len(&self) -> usize {
        self.clips.iter().map(|c| c.end()).max().unwrap_or(0)
    }

    /// Global timeline frame -> (clip index, local frame within its in/out).
    /// None = a GAP (or past the end): plays black.
    pub fn locate(&self, global: usize) -> Option<(usize, usize)> {
        for (i, c) in self.clips.iter().enumerate() {
            if global >= c.start && global < c.end() {
                return Some((i, global - c.start));
            }
        }
        None
    }

    /// A hash of everything the timeline and bin AUTHOR — clips, trims,
    /// placements, versions, inputs, names, folders, fps, which shot each
    /// clip plays — and nothing the shots hold (their graphs have wobble's
    /// own undo). Two projects with the same signature are the same edit,
    /// so a change in it is one history step, and a difference from the
    /// saved one is unsaved work.
    pub fn signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.fps.to_bits().hash(&mut h);
        self.folders.hash(&mut h);
        for (tag, list) in [(0u8, &self.clips), (1, &self.bin)] {
            tag.hash(&mut h);
            list.len().hash(&mut h);
            for c in list {
                (c.uid, c.current, c.in_off, c.out_off, c.start, c.shot, c.master).hash(&mut h);
                (&c.name, &c.folder, c.input.map(input_label)).hash(&mut h);
                c.identity().hash(&mut h);
                c.versions.len().hash(&mut h);
            }
        }
        h.finish()
    }

    /// Keep clips ordered by position; returns the new index of `uid`.
    pub fn sort_timeline(&mut self, uid: u64) -> Option<usize> {
        self.clips.sort_by_key(|c| c.start);
        self.clips.iter().position(|c| c.uid == uid)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let spec = |c: &Clip| ClipSpec {
            sample: match c.media() {
                Media::Frames(f) => f[c.in_off].clone(),
                Media::Movie { path, .. } => path.clone(),
                Media::Blank { tag } => tag.clone(),
            },
            blank: c.is_blank(),
            name: c.is_blank().then(|| c.name.clone()),
            in_off: c.in_off,
            out_off: c.out_off,
            input: c.input.map(|cs| input_label(cs)),
            start: Some(c.start),
            folder: c.folder.clone(),
            master: c.master,
            shot: c.shot.map(serde_json::Value::from),
        };
        let file = ProjectFile {
            fps: self.fps,
            clips: self.clips.iter().map(spec).collect(),
            bin: self.bin.iter().map(spec).collect(),
            shots: self.shots.clone(),
            folders: self.folders.clone(),
        };
        std::fs::write(path, serde_json::to_string_pretty(&file)?)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let file: ProjectFile = serde_json::from_str(
            &std::fs::read_to_string(path).with_context(|| path.display().to_string())?,
        )?;
        let revive = |spec: &ClipSpec| -> Result<Clip> {
            let mut clip = if spec.blank {
                let mut c = Clip::blank_tagged(spec.sample.clone());
                c.name = spec.name.clone().unwrap_or_else(|| "blank".into());
                c
            } else {
                Clip::from_sample(&spec.sample)?
            };
            let n = clip.media().len().max(1);
            clip.in_off = spec.in_off.min(n - 1);
            clip.out_off = spec.out_off.min(n - 1).max(clip.in_off);
            clip.input = spec.input.as_deref().and_then(input_by_label);
            clip.folder = spec.folder.clone();
            clip.master = spec.master;
            match &spec.shot {
                Some(serde_json::Value::Number(n)) => clip.shot = n.as_u64(),
                Some(serde_json::Value::Object(o)) => {
                    clip.legacy_script =
                        o.get("script").and_then(|s| s.as_str()).map(PathBuf::from);
                }
                _ => {}
            }
            Ok(clip)
        };
        let mut clips = Vec::new();
        for spec in &file.clips {
            let mut c = revive(spec)?;
            // Older files carry no positions: pack butt-joined.
            c.start = spec
                .start
                .unwrap_or_else(|| clips.last().map(|p: &Clip| p.end()).unwrap_or(0));
            clips.push(c);
        }
        clips.sort_by_key(|c| c.start);
        let mut bin = Vec::new();
        for spec in &file.bin {
            bin.push(revive(spec)?);
        }
        let mut folders = file.folders;
        for c in &bin {
            if !c.folder.is_empty() && !folders.contains(&c.folder) {
                folders.push(c.folder.clone());
            }
        }
        Ok(Self { fps: file.fps, clips, bin, shots: file.shots, folders })
    }

    /// Store a shot document under a fresh id. Ids are small and sequential
    /// on purpose: the app namespaces each shot's node ids by its shot id,
    /// so footage loaded for one shot is never mistaken for another's.
    #[allow(dead_code)]
    pub fn mint_shot(&mut self, doc: serde_json::Value) -> u64 {
        let id = self.shots.keys().next_back().map_or(1, |k| k + 1);
        self.shots.insert(id, doc);
        id
    }
}

/// Choose a start for a clip of `len` frames wanting `desired`, never
/// overlapping the existing clips (element `skip` excluded — a clip being
/// MOVED doesn't collide with itself). A collision snaps to the blocking
/// clip's nearer side; repeated until stable, so a drop into a too-small
/// slot butt-joins instead of overlapping.
pub fn resolve_start(clips: &[Clip], len: usize, desired: i64, skip: Option<usize>) -> usize {
    let mut s = desired.max(0) as usize;
    for _ in 0..=clips.len() {
        let mut moved = false;
        for (i, c) in clips.iter().enumerate() {
            if Some(i) == skip {
                continue;
            }
            let (cs, ce) = (c.start, c.end());
            if s < ce && s + len > cs {
                let after = ce;
                if cs >= len {
                    let before = cs - len;
                    let d_before = (desired - before as i64).abs();
                    let d_after = (after as i64 - desired).abs();
                    s = if d_before <= d_after { before } else { after };
                } else {
                    s = after;
                }
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    s
}

/// Preset label for a colourspace (project files, combo text).
pub fn input_label(cs: Colorspace) -> String {
    Colorspace::PRESETS
        .iter()
        .find(|(_, c)| *c == cs)
        .map(|(l, _)| l.to_string())
        .unwrap_or_else(|| cs.label())
}

pub fn input_by_label(label: &str) -> Option<Colorspace> {
    Colorspace::PRESETS
        .iter()
        .find(|(l, _)| *l == label)
        .map(|(_, c)| *c)
}

#[derive(Serialize, Deserialize)]
struct ProjectFile {
    fps: f32,
    clips: Vec<ClipSpec>,
    #[serde(default)]
    bin: Vec<ClipSpec>,
    /// Embedded shot documents by id: each is a complete wobble setup
    /// document, byte for byte what a standalone script contains, so
    /// extracting one is a file write, not a conversion.
    #[serde(default)]
    shots: std::collections::BTreeMap<u64, serde_json::Value>,
    #[serde(default)]
    folders: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct ClipSpec {
    /// The media's sample frame; a blank clip's tag.
    sample: PathBuf,
    /// No footage: `sample` is a tag, the shot is the whole clip.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    blank: bool,
    /// A blank clip's name (footage clips take theirs from the files).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    in_off: usize,
    out_off: usize,
    #[serde(default)]
    input: Option<String>,
    #[serde(default)]
    start: Option<usize>,
    #[serde(default)]
    folder: String,
    #[serde(default)]
    master: Option<u64>,
    /// A shot id — or, in files from the standalone app, an object carrying
    /// the script path (`{"script": …}`), imported on open.
    #[serde(default)]
    shot: Option<serde_json::Value>,
}

/// Find the LAST `v<digits>` token in a name. Returns (start, end_exclusive).
fn version_span(name: &str) -> Option<(usize, usize)> {
    let b = name.as_bytes();
    let mut found = None;
    let mut i = 0;
    while i < b.len() {
        if (b[i] == b'v' || b[i] == b'V') && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
            // Must not be preceded by a letter or digit ("previz" is not a version).
            let ok = i == 0 || !b[i - 1].is_ascii_alphanumeric();
            let mut j = i + 1;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if ok {
                found = Some((i, j));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    found
}

fn seq_media(seq: &sequitur::FileSequence) -> Media {
    Media::Frames(seq.items().iter().map(|it| it.path()).collect())
}

/// Discover every version of the sequence `sample` belongs to.
///
/// Strategy: prefer a v-token in the file prefix (siblings in the same dir);
/// otherwise a v-token in the parent directory name (sibling dirs holding the
/// same sequence name); otherwise a single unversioned entry.
fn scan_seq_versions(sample: &Path) -> Result<(String, Vec<Version>)> {
    let item = sequitur::Item::from_path(sample)
        .ok_or_else(|| anyhow!("not a numbered sequence: {}", sample.display()))?;
    let dir = sample
        .parent()
        .ok_or_else(|| anyhow!("no parent dir"))?
        .to_path_buf();
    let prefix = item.prefix().to_string();
    let ext = item.extension().to_string();
    let name = prefix.trim_end_matches(['.', '_']).to_string();

    // Case 1: version token in the filename prefix.
    if let Some((s, e)) = version_span(&prefix) {
        let (pre, post) = (&prefix[..s], &prefix[e..]);
        let listing = sequitur::FileSequence::parse_directory(&dir, 1)
            .map_err(|e| anyhow!("scan {}: {e:?}", dir.display()))?;
        let mut out = Vec::new();
        for seq in &listing.sequences {
            let it = &seq.items()[0];
            if it.extension() != ext {
                continue;
            }
            let p = it.prefix();
            if let Some((s2, e2)) = version_span(p) {
                if &p[..s2] == pre && &p[e2..] == post {
                    out.push(Version {
                        label: p[s2..e2].to_string(),
                        number: p[s2 + 1..e2].parse().unwrap_or(0),
                        media: seq_media(seq),
                    });
                }
            }
        }
        if !out.is_empty() {
            out.sort_by_key(|v| v.number);
            return Ok((name, out));
        }
    }

    // Case 2: version token in the parent directory name.
    let dir_name = dir.file_name().map(|n| n.to_string_lossy().into_owned());
    if let (Some(dn), Some(grand)) = (dir_name, dir.parent()) {
        if let Some((s, e)) = version_span(&dn) {
            let (pre, post) = (&dn[..s], &dn[e..]);
            let mut out = Vec::new();
            if let Ok(rd) = std::fs::read_dir(grand) {
                for entry in rd.flatten() {
                    if !entry.path().is_dir() {
                        continue;
                    }
                    let vname = entry.file_name().to_string_lossy().into_owned();
                    if let Some((s2, e2)) = version_span(&vname) {
                        if &vname[..s2] == pre && &vname[e2..] == post {
                            if let Ok(listing) =
                                sequitur::FileSequence::parse_directory(&entry.path(), 1)
                            {
                                for seq in &listing.sequences {
                                    let it = &seq.items()[0];
                                    if it.prefix() == prefix && it.extension() == ext {
                                        out.push(Version {
                                            label: vname[s2..e2].to_string(),
                                            number: vname[s2 + 1..e2].parse().unwrap_or(0),
                                            media: seq_media(seq),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !out.is_empty() {
                out.sort_by_key(|v| v.number);
                return Ok((name, out));
            }
        }
    }

    // Case 3: unversioned.
    let listing = sequitur::FileSequence::parse_directory(&dir, 1)
        .map_err(|e| anyhow!("scan {}: {e:?}", dir.display()))?;
    let seq = listing
        .sequences
        .iter()
        .find(|s| {
            let it = &s.items()[0];
            it.prefix() == prefix && it.extension() == ext
        })
        .ok_or_else(|| anyhow!("sequence not found for {}", sample.display()))?;
    Ok((
        name,
        vec![Version { label: "—".into(), number: 0, media: seq_media(seq) }],
    ))
}

/// Discover every version of the movie `sample` belongs to: a v-token in the
/// file stem (siblings in the same dir), else in the parent directory name,
/// else a single unversioned entry. Each candidate is header-probed for its
/// frame count.
fn scan_movie_versions(sample: &Path) -> Result<(String, Vec<Version>)> {
    let decoder = AnyDecoder::new(true);
    let probe = |p: &Path| -> Option<usize> {
        match decoder.frame_count(p) {
            Ok(n) if n > 0 => Some(n as usize),
            Ok(_) => None,
            Err(e) => {
                log::warn!("probe {}: {e:#}", p.display());
                None
            }
        }
    };
    let dir = sample
        .parent()
        .ok_or_else(|| anyhow!("no parent dir"))?
        .to_path_buf();
    let stem = sample
        .file_stem()
        .ok_or_else(|| anyhow!("no file name"))?
        .to_string_lossy()
        .into_owned();
    let ext = sample
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_ascii_lowercase();

    // Case 1: version token in the file stem.
    if let Some((s, e)) = version_span(&stem) {
        let (pre, post) = (&stem[..s], &stem[e..]);
        let name = pre.trim_end_matches(['.', '_', '-']).to_string();
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if !p.is_file()
                    || p.extension()
                        .map(|x| x.to_string_lossy().to_ascii_lowercase())
                        != Some(ext.clone())
                {
                    continue;
                }
                let st = p.file_stem().unwrap_or_default().to_string_lossy().into_owned();
                if let Some((s2, e2)) = version_span(&st) {
                    if &st[..s2] == pre && &st[e2..] == post {
                        if let Some(len) = probe(&p) {
                            out.push(Version {
                                label: st[s2..e2].to_string(),
                                number: st[s2 + 1..e2].parse().unwrap_or(0),
                                media: Media::Movie { path: p, len },
                            });
                        }
                    }
                }
            }
        }
        if !out.is_empty() {
            out.sort_by_key(|v| v.number);
            return Ok((name, out));
        }
    }

    // Case 2: version token in the parent directory name.
    let file_name = sample.file_name().unwrap_or_default().to_owned();
    let dir_name = dir.file_name().map(|n| n.to_string_lossy().into_owned());
    if let (Some(dn), Some(grand)) = (dir_name, dir.parent()) {
        if let Some((s, e)) = version_span(&dn) {
            let (pre, post) = (&dn[..s], &dn[e..]);
            let mut out = Vec::new();
            if let Ok(rd) = std::fs::read_dir(grand) {
                for entry in rd.flatten() {
                    if !entry.path().is_dir() {
                        continue;
                    }
                    let vname = entry.file_name().to_string_lossy().into_owned();
                    if let Some((s2, e2)) = version_span(&vname) {
                        if &vname[..s2] == pre && &vname[e2..] == post {
                            let p = entry.path().join(&file_name);
                            if let Some(len) = p.is_file().then(|| probe(&p)).flatten() {
                                out.push(Version {
                                    label: vname[s2..e2].to_string(),
                                    number: vname[s2 + 1..e2].parse().unwrap_or(0),
                                    media: Media::Movie { path: p, len },
                                });
                            }
                        }
                    }
                }
            }
            if !out.is_empty() {
                out.sort_by_key(|v| v.number);
                return Ok((stem, out));
            }
        }
    }

    // Case 3: unversioned.
    let len = probe(sample).ok_or_else(|| anyhow!("unreadable movie: {}", sample.display()))?;
    Ok((
        stem,
        vec![Version {
            label: "—".into(),
            number: 0,
            media: Media::Movie { path: sample.to_path_buf(), len },
        }],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Project round trip: timeline trims, bin entries, version choices and
    /// input picks survive save -> load (which re-scans disk).
    #[test]
    fn project_roundtrip() {
        let a = Path::new("testdata/shotA/v001/frame.0000.exr");
        let b = Path::new("testdata/shotB/v002/frame.0000.exr");
        if !a.is_file() || !b.is_file() {
            eprintln!("testdata missing; skipping");
            return;
        }
        let mut p = Project::default();
        let mut tl = Clip::from_sample(a).unwrap();
        tl.in_off = 5;
        tl.out_off = 20;
        tl.input = Some(Colorspace::SRGB);
        tl.start = 7; // gap before the first clip
        p.clips.push(tl);
        p.bin.push(Clip::from_sample(b).unwrap());

        let dir = std::env::temp_dir().join("wobble_edit_test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("roundtrip.wedit");
        p.save(&f).unwrap();
        let q = Project::load(&f).unwrap();
        assert_eq!(q.clips.len(), 1);
        assert_eq!(q.bin.len(), 1);
        assert_eq!((q.clips[0].in_off, q.clips[0].out_off), (5, 20));
        assert_eq!(q.clips[0].version().label, p.clips[0].version().label);
        assert_eq!(q.bin[0].version().label, p.bin[0].version().label);
        assert_eq!(q.clips[0].len(), 16);
        assert_eq!(q.clips[0].input, Some(Colorspace::SRGB));
        assert_eq!(q.bin[0].input, None);
        assert_eq!(q.clips[0].start, 7, "timeline position lost in round trip");
    }

    /// A blank clip needs no files: it saves and loads with its name, trim,
    /// placement and shot, keeps its own identity through a duplicate, and
    /// can be trimmed far past its default length.
    #[test]
    fn a_blank_clip_round_trips_without_footage() {
        let mut p = Project::default();
        let mut tl = Clip::blank("title card", 48);
        tl.start = 10;
        tl.shot = Some(3);
        assert_eq!(tl.len(), 48);
        assert!(tl.media().len() > 10_000, "the out point has headroom");
        let dup = tl.duplicate_master();
        assert_ne!(dup.identity(), tl.identity(), "a duplicate is its own clip");
        assert_eq!(dup.len(), 48);
        p.clips.push(tl);
        p.bin.push(dup);
        let dir = std::env::temp_dir().join("wobble_edit_blank_test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("blank.wedit");
        p.save(&f).unwrap();
        let q = Project::load(&f).unwrap();
        assert!(q.clips[0].is_blank() && q.bin[0].is_blank());
        assert_eq!(q.clips[0].name, "title card");
        assert_eq!(q.bin[0].name, "title card copy");
        assert_eq!((q.clips[0].in_off, q.clips[0].out_off, q.clips[0].start), (0, 47, 10));
        assert_eq!(q.clips[0].shot, Some(3));
        assert_eq!(q.clips[0].identity(), p.clips[0].identity());
        assert_eq!(q.total_len(), 58);
    }

    /// Gap semantics: placement resolves overlaps but allows free positions;
    /// locate() reports gaps as None and the total spans them.
    #[test]
    fn gaps_place_and_locate() {
        let a = Path::new("testdata/shotA/v001/frame.0000.exr");
        if !a.is_file() {
            eprintln!("testdata missing; skipping");
            return;
        }
        let mut p = Project::default();
        let mut c1 = Clip::from_sample(a).unwrap(); // 48 frames
        c1.start = 0;
        p.clips.push(c1);
        // Wants 100: free — lands exactly there, leaving a 52-frame gap.
        let s = resolve_start(&p.clips, 48, 100, None);
        assert_eq!(s, 100);
        let mut c2 = Clip::from_sample(a).unwrap();
        c2.start = s;
        p.clips.push(c2);
        assert_eq!(p.total_len(), 148);
        assert_eq!(p.locate(10).map(|(i, _)| i), Some(0));
        assert_eq!(p.locate(60), None, "gap must locate as None");
        assert_eq!(p.locate(100), Some((1, 0)));
        // Wants 90: overlaps clip 2's head — snaps to its nearer side.
        assert_eq!(resolve_start(&p.clips, 48, 90, None), 52);
        // Moving clip 2 itself ignores its own footprint.
        assert_eq!(resolve_start(&p.clips, 48, 100, Some(1)), 100);
    }

    #[test]
    fn movie_versions_and_input_guess() {
        let mov = Path::new("testdata/movs/shotC_v001.mov");
        if !mov.is_file() {
            eprintln!("movie testdata missing; skipping");
            return;
        }
        let clip = Clip::from_sample(mov).unwrap();
        assert!(clip.versions.len() >= 2, "sibling versions not found");
        assert_eq!(clip.version().label, "v001");
        assert!(matches!(clip.media(), Media::Movie { len, .. } if *len > 0));
        assert_eq!(clip.effective_input(), Colorspace::REC709);

        let exr = Clip::from_sample(Path::new("testdata/shotA/v001/frame.0000.exr")).unwrap();
        assert_eq!(exr.effective_input(), Colorspace::ACESCG);
    }
}
