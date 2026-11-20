//! The clip browser for Load nodes: seq_manip's browser (`seq_manip_imfram`)
//! on imfram's pieces, over wobble's own decoders. Along the top, the path
//! bar (back / forward / up, rescan, favourite, the path with Tab
//! completion) and the view's controls. Down the left, home, the comp's folder, favourites, recent
//! folders, the folder tree and the bin. In the middle, imfram's gallery
//! (tiles) or list (rows, sortable columns) over the folder: numbered
//! frames collapse into one sequence item, versions of a shot into one
//! entry, a tile or a row's thumbnail scrubs under the pointer. Down the
//! right, what types are here and what to hide. Under the items, an
//! inspector for the one selected clip: format, size, missing frames and a
//! player. Space is a full-window quick look; typing a name jumps to it.

use crate::theme;
use egui::{Color32, RichText};
use engine::decode::ExrDecoder;
use half::f16;
use imfram::gallery::nav::{self, FolderTree, PathBar};
use imfram::gallery::typeahead::TypeAhead;
use imfram::gallery::{Details, Dragged, Gallery, GalleryEvent, Items, List, Request};
use imfram::table::Column;
use sequitur::{EntryKind, FileSequence};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::SystemTime; // file mtimes: std's type, not web_time's


pub enum BrowserAction {
    None,
    /// User committed a sequence: display pattern + resolved frame paths +
    /// missing-frame count (for the Read properties readout).
    Pick { display: String, frames: Vec<PathBuf>, missing: u64 },
    /// The bin was committed: build a Read node per entry.
    PickMany(Vec<Picked>),
    /// A folder was chosen (`SeqBrowser::pick_folder`): the directory shown.
    PickDir(PathBuf),
    Close,
}

/// One sequence collected in the bin.
#[derive(Clone, PartialEq)]
pub struct Picked {
    pub display: String,
    pub frames: Vec<PathBuf>,
    pub missing: u64,
}

/// In a page there is no disk to list: what to do instead.
#[cfg(target_arch = "wasm32")]
fn web_hints(ui: &mut egui::Ui) {
    ui.add_space(8.0);
    ui.label(RichText::new("The web version can't browse your disk.").strong());
    ui.label("A browser only lets the page see files you give it. To load footage:");
    ui.add_space(4.0);
    ui.label("·  drag files or a whole folder anywhere onto the page; each becomes a Load");
    ui.label("·  or pick files with the button below (or open, in the menu bar)");
    ui.label("·  for an image sequence, drop its FOLDER, or select all of its frames at once");
    ui.add_space(4.0);
    ui.weak("EXR, DPX, PNG, JPEG and MP4/MOV movies (H.264, HEVC, AV1, VP9; not ProRes) load here; setups (.wbl) open the same way.");
    ui.weak("The desktop app browses folders and finds a sequence from any one frame.");
    ui.add_space(8.0);
    if ui.button("choose files…").clicked() {
        if let Err(e) = crate::webmain::pick() {
            ui.colored_label(theme::error(), format!("{e:#}"));
        }
    }
}

/// The listing's BASE gate: decodable media only, or the whole directory.
/// Individual extensions are then unticked in the types sidebar
/// (`SeqBrowser::excluded`); `shown` puts the two together.
#[derive(Clone, Copy, PartialEq)]
pub enum TypeFilter {
    /// Formats the compositor can actually read.
    Supported,
    All,
}

impl TypeFilter {
    fn label(&self) -> &'static str {
        match self {
            // NOT a list: `SUPPORTED_EXTS` is eleven entries with movies
            // enabled. What it IS is spelled out in `supported_hint()`, on
            // hover, where there is room to be exact — and where it can say
            // whether this build has movie support. The sidebar's per-type
            // ticks are the list, with counts.
            TypeFilter::Supported => "supported media",
            TypeFilter::All => "all files",
        }
    }

    fn accepts(&self, ext: Option<&str>) -> bool {
        match self {
            TypeFilter::Supported => ext.is_some_and(is_supported_ext),
            TypeFilter::All => true,
        }
    }
}

fn is_supported_ext(e: &str) -> bool {
    SUPPORTED.iter().any(|s| e.eq_ignore_ascii_case(s))
}

/// Extensions are keyed lower-case, and a file with none keys as "".
fn ext_key(ext: Option<&str>) -> String {
    ext.unwrap_or("").to_lowercase()
}

/// Does the type gate let a file with this extension through? The base
/// filter first, then the sidebar's unticked extensions — so unticking
/// `.png` hides pngs under "all files" and "supported media" alike, in the
/// flat listing and the recursive search alike.
fn shown(types: TypeFilter, excluded: &BTreeSet<String>, ext: Option<&str>) -> bool {
    types.accepts(ext) && !excluded.contains(&ext_key(ext))
}

/// What the action bar's buttons did this frame.
#[derive(Default)]
struct ActionRow {
    /// "add N selected to bin"
    add: bool,
    /// "create N reads"
    commit: bool,
}

/// The action bar: the selection's "add to bin" on the left, the bin's
/// commit on the right, one row.
///
/// The `ui.horizontal` wrapper is load-bearing, not tidiness. A bare
/// `with_layout(right_to_left(Align::Center))` claims the FULL available
/// height in order to centre within it — and inside a content-sized bottom
/// panel that height is fed by the panel's own size from the previous
/// frame. The two ratchet each other: the panel grew ~7pt per frame until
/// it had swallowed the window and the listing was a sliver. `horizontal`
/// constrains the row to one line, which breaks the loop.
/// `browser::action_bar_tests` holds it to that.
fn action_row(ui: &mut egui::Ui, selected: usize, binned: usize) -> ActionRow {
    let mut out = ActionRow::default();
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if selected > 0 {
            let label = RichText::new(format!(
                "add {selected} selected to bin"
            ))
            .size(14.0);
            out.add = ui
                .add(
                    egui::Button::new(label)
                        .corner_radius(6.0)
                        .min_size(egui::vec2(170.0, 30.0)),
                )
                .on_hover_text(
                    "click selects one · shift-click a range · ⌘/ctrl-click toggles · ⌘A all",
                )
                .clicked();
        }
        if binned > 0 {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let accent = theme::browser_accent();
                let label = RichText::new(format!("create {binned} reads"))
                    .strong()
                    .color(theme::text_hot())
                    .size(14.0);
                out.commit = ui
                    .add(
                        egui::Button::new(label)
                            .fill(accent)
                            .corner_radius(6.0)
                            .min_size(egui::vec2(170.0, 30.0)),
                    )
                    .on_hover_text("one Load node per binned sequence, then close")
                    .clicked();
            });
        }
    });
    ui.add_space(4.0);
    out
}


/// Extensions the decoder handles (shared with the engine, so the filter
/// and the loader always agree).
use engine::decode::SUPPORTED_EXTS as SUPPORTED;

/// What "supported media" actually covers, read off the decoder's own list
/// so it can never drift from what will load. Movie extensions are absent
/// from that list in a build without the `movie` feature, and the hint
/// says so rather than leaving the user to guess why their .mov is hidden.
fn supported_hint() -> String {
    let exts = SUPPORTED
        .iter()
        .map(|e| format!(".{e}"))
        .collect::<Vec<_>>()
        .join("  ");
    let movies = if engine::decode::MOVIE_EXTS.is_empty() {
        "\n\nMovies (ProRes, H.264) need a build with the `movie` feature; \
         this one does not have it, so they are not listed even under \
         \"all files\" as loadable."
    } else {
        "\n\nMovies are one file holding many frames: they list as a single \
         clip, never folded into a numbered sequence."
    };
    format!("Shows only what this build can decode:\n{exts}{movies}")
}

/// The quick look's frames: a bounded texture cache fed by a small
/// pool of decode workers, refilled from whatever the UI asked for THIS
/// frame — so scrolling (or scrubbing) chases what is on screen and
/// abandons what scrolled away.
struct Thumbs {
    tex: HashMap<FrameKey, egui::TextureHandle>,
    /// Insertion order, for eviction.
    order: Vec<FrameKey>,
    inflight: Vec<FrameKey>,
    wanted: Vec<FrameKey>,
    tx: Sender<(FrameKey, egui::ColorImage)>,
    rx: Receiver<(FrameKey, egui::ColorImage)>,
    /// Width the decode is downsampled to, and how many textures to hold.
    max_w: usize,
    cap: usize,
}

const THUMB_W: usize = 320;
const THUMB_WORKERS: usize = 4;
/// The quick-look cache: far fewer frames, decoded far larger.
const BIG_W: usize = 1600;
const BIG_CAP: usize = 32;

impl Thumbs {
    fn new(max_w: usize, cap: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            tex: HashMap::new(),
            order: Vec::new(),
            inflight: Vec::new(),
            wanted: Vec::new(),
            tx,
            rx,
            max_w,
            cap,
        }
    }

    /// The texture if it is ready; otherwise note that it is wanted.
    fn get(&mut self, key: &FrameKey) -> Option<egui::TextureHandle> {
        if let Some(t) = self.tex.get(key) {
            return Some(t.clone());
        }
        if !self.wanted.contains(key) {
            self.wanted.push(key.clone());
        }
        None
    }

    fn pending(&self, key: &FrameKey) -> bool {
        self.inflight.contains(key)
    }

    /// The nearest already-decoded frame of the same source, if any.
    ///
    /// Scrubbing outruns the decoder, and a blank tile mid-drag reads as a
    /// broken sequence rather than a slow one. Holding the closest frame we
    /// do have keeps the picture moving and simply gets crisper when the
    /// wanted one lands.
    fn nearest(&self, key: &FrameKey) -> Option<egui::TextureHandle> {
        self.tex
            .iter()
            .filter(|((p, _), _)| *p == key.0)
            .min_by_key(|((_, f), _)| f.abs_diff(key.1))
            .map(|(_, t)| t.clone())
    }

    /// Collect finished decodes, then start more from this frame's wants.
    fn pump(&mut self, ctx: &egui::Context, decoder: &Arc<dyn ExrDecoder>) {
        while let Ok((key, img)) = self.rx.try_recv() {
            self.inflight.retain(|k| *k != key);
            let tex = ctx.load_texture("seq-thumb", img, egui::TextureOptions::LINEAR);
            if self.tex.insert(key.clone(), tex).is_none() {
                self.order.push(key);
            }
            while self.order.len() > self.cap {
                let old = self.order.remove(0);
                self.tex.remove(&old);
            }
        }
        // Newest wants first: while scrubbing, the frame asked for THIS
        // frame matters and the one from three frames ago does not.
        let mut wanted = std::mem::take(&mut self.wanted);
        wanted.reverse();
        for key in wanted {
            if self.inflight.len() >= THUMB_WORKERS {
                break;
            }
            if self.tex.contains_key(&key) || self.pending(&key) {
                continue;
            }
            self.inflight.push(key.clone());
            spawn_thumb(decoder.clone(), key, self.max_w, self.tx.clone());
        }
    }
}

/// Preview playback rate. Not the clip's own rate — the browser has no
/// authority on that (a sequence has none at all, and a movie's is only
/// known after a header read) — so this is an honest fixed shuttle speed,
/// labelled in the UI as what it is.
const PREVIEW_FPS: f32 = 24.0;

struct Preview {
    started: web_time::Instant,
    /// Frame the transport was sitting on when play was pressed.
    from: usize,
    len: usize,
}

impl Preview {
    /// Where the transport is now, derived from wall time.
    fn frame(&self) -> usize {
        let elapsed = self.started.elapsed().as_secs_f32();
        (self.from + (elapsed * PREVIEW_FPS) as usize) % self.len.max(1)
    }
}


fn fit_rect(rect: egui::Rect, aspect: f32) -> egui::Rect {
    let ba = rect.width() / rect.height().max(1.0);
    if aspect > ba {
        egui::Rect::from_center_size(rect.center(), egui::vec2(rect.width(), rect.width() / aspect))
    } else {
        egui::Rect::from_center_size(rect.center(), egui::vec2(rect.height() * aspect, rect.height()))
    }
}

/// Draw a thumbnail (or a placeholder) fitted into `rect`.

pub type FrameKey = (PathBuf, u64);

#[derive(Clone)]
pub struct SeqRow {
    /// `name.####.exr` (or a bare filename for a single frame)
    pub pattern: String,
    pub frames: Arc<[PathBuf]>,
    pub numbers: Vec<i32>,
    pub first: i32,
    pub last: i32,
    pub missing: u64,
    pub readable: bool,
    /// Lower-case extension, for the type filter.
    pub ext: Option<String>,
    /// Newest mtime across the sequence's ends (None = could not stat).
    pub modified: Option<SystemTime>,
    /// One container holding many frames, rather than a file per frame.
    pub movie: bool,
}

impl SeqRow {
    /// Length on a timeline, in frames. The SPAN, not the file count, so a
    /// gappy sequence still reports the duration it would occupy.
    pub fn duration(&self) -> usize {
        (self.last as i64 - self.first as i64 + 1).max(1) as usize
    }

    /// How many frames are scrubbable. A movie's length is not knowable
    /// from the listing — it takes a header read — so the caller passes
    /// what `Probe` found; everything else counts its own files.
    pub fn scrub_len(&self, probed: u64) -> usize {
        if self.movie {
            (probed as usize).max(1)
        } else {
            self.frames.len()
        }
    }

    /// The i-th scrubbable frame.
    pub fn frame_key(&self, i: usize) -> FrameKey {
        if self.movie {
            (self.frames[0].clone(), i as u64)
        } else {
            (self.frames[i.min(self.frames.len().saturating_sub(1))].clone(), 0)
        }
    }
}

/// Which column the listing is ordered by. Shared by the flat listing and
/// the recursive search, so a sort survives switching between them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SortKey {
    Name,
    Date,
    Duration,
}

impl SortKey {
    fn label(&self) -> &'static str {
        match self {
            SortKey::Name => "name",
            SortKey::Date => "date",
            SortKey::Duration => "duration",
        }
    }

    /// Names read best A-Z; dates and durations best biggest-first.
    fn desc_by_default(&self) -> bool {
        !matches!(self, SortKey::Name)
    }
}


/// Order two rows by the active column, name-breaking ties so the listing
/// never reshuffles between frames.
fn row_cmp(a: &SeqRow, b: &SeqRow, key: SortKey, desc: bool) -> std::cmp::Ordering {
    let ord = match key {
        SortKey::Name => natural_cmp(&a.pattern, &b.pattern),
        // None (unstat-able) sorts oldest.
        SortKey::Date => a
            .modified
            .cmp(&b.modified)
            .then_with(|| natural_cmp(&a.pattern, &b.pattern)),
        SortKey::Duration => a
            .duration()
            .cmp(&b.duration())
            .then_with(|| natural_cmp(&a.pattern, &b.pattern)),
    };
    if desc {
        ord.reverse()
    } else {
        ord
    }
}

/// Names compared the way people read them: case folded, and a run of
/// digits compared as a NUMBER, so `shot2` comes before `shot10`. A tie
/// (`Shot` vs `shot`, `v1` vs `v01`) falls back to the raw bytes so the
/// order is total and never reshuffles.
pub fn natural_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let (mut x, mut y) = (a.chars().peekable(), b.chars().peekable());
    loop {
        match (x.peek().copied(), y.peek().copied()) {
            (None, None) => return a.cmp(b),
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(c), Some(d)) if c.is_ascii_digit() && d.is_ascii_digit() => {
                let run = |it: &mut std::iter::Peekable<std::str::Chars>| {
                    let mut s = String::new();
                    while let Some(c) = it.peek().copied().filter(char::is_ascii_digit) {
                        s.push(c);
                        it.next();
                    }
                    s
                };
                let (p, q) = (run(&mut x), run(&mut y));
                // Leading zeros carry no value; then the longer run is the
                // bigger number, and equal lengths compare digit by digit.
                let (p0, q0) = (p.trim_start_matches('0'), q.trim_start_matches('0'));
                let ord = p0.len().cmp(&q0.len()).then_with(|| p0.cmp(q0));
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            (Some(c), Some(d)) => {
                let ord = c.to_lowercase().cmp(d.to_lowercase());
                if ord != Ordering::Equal {
                    return ord;
                }
                x.next();
                y.next();
            }
        }
    }
}

/// The version tokens in a path: every `v###` (or `V###`) that starts a
/// word — after a separator, not inside one (`dev12` is not a version) —
/// and ends at a non-digit. Returns the LAST one's number and the path
/// with EVERY token replaced, which is what all versions of the same shot
/// have in common. Pipelines often say it twice, in a folder and in the
/// file (`pass/v051/sh030_pass_v051.1000.exr`), and both must go or each
/// version keys apart. Either separator: paths come from Windows too.
pub fn version_of(path: &str) -> Option<(u32, String)> {
    let b = path.as_bytes();
    let mut key = String::with_capacity(path.len());
    let (mut last, mut copied, mut i) = (None, 0, 0);
    while i < b.len() {
        if matches!(b[i], b'v' | b'V') && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
            let end = (i + 1..b.len()).find(|&j| !b[j].is_ascii_digit()).unwrap_or(b.len());
            if (2..=7).contains(&(end - i)) {
                if let Ok(n) = path[i + 1..end].parse::<u32>() {
                    key.push_str(&path[copied..i]);
                    key.push_str("v#");
                    copied = end;
                    last = Some(n);
                    i = end;
                    continue;
                }
            }
        }
        i += 1;
    }
    key.push_str(&path[copied..]);
    last.map(|n| (n, key))
}

/// A folder that IS a version (`v051`): its renders are listed in its
/// parent, so browsing a pass shows the pass, not a column of folders.
fn is_version_dir(p: &Path) -> bool {
    p.file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| version_of(n).filter(|(_, k)| k == "v#"))
        .is_some()
}

/// The versions of one shot, as the collapsed listing shows them: every
/// version present (ascending) and the one on show.
#[derive(Clone, Debug, PartialEq)]
pub struct VersionGroup {
    pub key: String,
    pub versions: Vec<u32>,
    pub shown: u32,
}

/// Collapse versions of the same shot to one entry. `items` are (index,
/// full path) of the rows the listing would draw; the answer is the
/// indices to HIDE, and for each shown row that stands for several
/// versions, its group. A group shows the version picked for it, if that
/// version is still listed, else the latest. Rows with no version token,
/// or alone in their group, are untouched.
pub fn collapse_versions(
    items: impl IntoIterator<Item = (usize, String)>,
    picks: &HashMap<String, u32>,
) -> (std::collections::HashSet<usize>, HashMap<usize, VersionGroup>) {
    let mut groups: BTreeMap<String, Vec<(u32, usize)>> = BTreeMap::new();
    for (i, path) in items {
        if let Some((v, key)) = version_of(&path) {
            groups.entry(key).or_default().push((v, i));
        }
    }
    let mut hidden = std::collections::HashSet::new();
    let mut shown = HashMap::new();
    for (key, mut members) in groups {
        if members.len() < 2 {
            continue;
        }
        members.sort();
        let latest = members.last().unwrap().0;
        let want = picks.get(&key).copied().unwrap_or(latest);
        let pick = members.iter().find(|(v, _)| *v == want).map_or(latest, |m| m.0);
        let mut versions: Vec<u32> = members.iter().map(|m| m.0).collect();
        versions.dedup();
        let mut placed = false;
        for (v, i) in &members {
            if *v == pick && !placed {
                placed = true;
                shown.insert(*i, VersionGroup { key: key.clone(), versions: versions.clone(), shown: pick });
            } else {
                hidden.insert(*i);
            }
        }
    }
    (hidden, shown)
}

/// The version picker on a collapsed row: `v003 ▾`, opening onto every
/// version of the shot. Returns the version picked this frame.
fn version_menu(ui: &mut egui::Ui, id: impl std::hash::Hash + std::fmt::Debug, g: &VersionGroup) -> Option<u32> {
    let mut picked = None;
    let latest = *g.versions.last().unwrap_or(&g.shown);
    let text = format!("v{:03}", g.shown);
    let text = if g.shown == latest {
        RichText::new(text).small()
    } else {
        // An older version on show must not pass for the current one.
        RichText::new(text).small().color(theme::warn())
    };
    egui::ComboBox::from_id_salt(id)
        .selected_text(text)
        .width(54.0)
        .show_ui(ui, |ui| {
            for &v in g.versions.iter().rev() {
                let label = if v == latest { format!("v{v:03}  latest") } else { format!("v{v:03}") };
                if ui.selectable_label(v == g.shown, label).clicked() {
                    picked = Some(v);
                }
            }
        })
        .response
        .on_hover_text(format!("{} versions of this shot", g.versions.len()));
    picked
}

/// Unix seconds, negative before the epoch.
fn unix_secs(t: SystemTime) -> i64 {
    match t.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(e) => -(e.duration().as_secs() as i64),
    }
}

/// Unix seconds -> (year, month, day, hour, minute), UTC. Hinnant's
/// civil-from-days; the alternative is a date crate for two columns.
fn civil(secs: i64) -> (i64, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    (if m <= 2 { y + 1 } else { y }, m, d, (rem / 3600) as u32, (rem % 3600 / 60) as u32)
}

/// Day resolution in the column: enough to scan, and small enough that the
/// UTC/local difference can only ever be a day at the edges.
fn fmt_date(t: Option<SystemTime>) -> String {
    match t {
        Some(t) => {
            let (y, m, d, _, _) = civil(unix_secs(t));
            format!("{y:04}-{m:02}-{d:02}")
        }
        None => "—".into(),
    }
}

/// The full stamp, for the hover. Marked UTC because it is.
fn fmt_date_full(t: Option<SystemTime>) -> String {
    match t {
        Some(t) => {
            let (y, m, d, hh, mm) = civil(unix_secs(t));
            format!("modified {y:04}-{m:02}-{d:02} {hh:02}:{mm:02} UTC")
        }
        None => "modification time unavailable".into(),
    }
}


fn fmt_frames(n: usize) -> String {
    format!("{n}f")
}

/// A bare drive letter (`C:`) as its ROOT (`C:/`). On Windows `C:` alone
/// means that drive's current directory, not its root, so a browser that
/// shows or navigates to one lands somewhere else. A pure string rule, so
/// it is tested on every OS; anything else passes through untouched.
fn drive_root(p: &Path) -> PathBuf {
    match p.to_str().map(str::trim) {
        Some(s) if s.len() == 2 && s.as_bytes()[0].is_ascii_alphabetic() && s.ends_with(':') => {
            PathBuf::from(format!("{s}/"))
        }
        _ => p.to_path_buf(),
    }
}

/// Newest mtime at either end of the sequence — two stats, not one per
/// frame, so a 5000-frame row still costs nothing to list.
fn seq_modified(frames: &[PathBuf]) -> Option<SystemTime> {
    let stat = |p: &PathBuf| std::fs::metadata(p).ok()?.modified().ok();
    let first = frames.first().and_then(stat);
    let last = frames.last().and_then(stat);
    first.max(last)
}

struct Listing {
    dirs: Vec<PathBuf>,
    seqs: Vec<SeqRow>,
    /// Loose files the decoder can't read; shown dimmed when the type
    /// filter asks for them.
    other: Vec<PathBuf>,
    /// `seqs[lifted..]` live in version folders (`v051/…`) one level down:
    /// listed while versions collapse, with those folders hidden, so a
    /// pass reads as one render with a version menu.
    lifted: usize,
    version_dirs: BTreeSet<PathBuf>,
}

impl Listing {
    /// Is row `i` listed, and folder `d`, under the versions toggle?
    fn row_on(&self, i: usize, collapse: bool) -> bool {
        collapse || i < self.lifted
    }
    fn dir_on(&self, d: &Path, collapse: bool) -> bool {
        !(collapse && self.version_dirs.contains(d))
    }
}

/// Lazily-computed per-sequence facts for the inspector.
struct Probe {
    format: String,
    total_bytes: u64,
    /// Frames inside the FILE — >1 only for a movie. Read once, here,
    /// because it costs a header parse and the scrub needs it every frame.
    frames: u64,
}

/// A recursive scan running on a worker: results stream in, and a
/// generation stamp drops anything from a superseded search.
struct Search {
    rx: Receiver<(u64, PathBuf, Vec<SeqRow>, Vec<String>)>,
    generation: u64,
    /// Directories walked so far, and whether the walk is still running.
    scanned: usize,
    running: bool,
    /// (directory, sequences) in discovery order.
    hits: Vec<(PathBuf, SeqRow)>,
    /// Loose files the walk saw but does not list (nothing decodes them),
    /// counted per extension so the sidebar can still say they are there.
    other: BTreeMap<String, usize>,
}

/// Pixel size per source, read off the file header on its own thread, so
/// a resolution floor can filter a directory without stalling the listing.
/// Nothing is read until the floor is non-zero; a row stays listed until
/// its size is known, then drops out if it is too small.
struct Dims {
    /// None = the header could not be read.
    known: HashMap<PathBuf, Option<(u32, u32)>>,
    /// Frames in a movie file (movies only), read with its header.
    counts: HashMap<PathBuf, u64>,
    inflight: std::collections::HashSet<PathBuf>,
    req: Sender<PathBuf>,
    res: Receiver<(PathBuf, Option<(u32, u32)>, Option<u64>)>,
}

impl Dims {
    fn new(decoder: Arc<dyn ExrDecoder>) -> Self {
        let (req, req_rx) = std::sync::mpsc::channel::<PathBuf>();
        let (res_tx, res) = std::sync::mpsc::channel();
        // One thread, sequential: a header is a few KB, and this must never
        // compete with the player's decode pool. It ends when the browser
        // drops its sender. Not in a page: this loop never ends, and there
        // it would hold the only thread (bg.rs) — dims simply stay unknown.
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            while let Ok(path) = req_rx.recv() {
                let dims = decoder.read_header(&path).ok().map(|i| (i.width, i.height));
                let frames = (dims.is_some() && engine::decode::is_movie(&path)).then(|| decoder.frame_count(&path).unwrap_or(1));
                if res_tx.send((path, dims, frames)).is_err() {
                    return;
                }
            }
        });
        #[cfg(target_arch = "wasm32")]
        drop((req_rx, res_tx, decoder));
        Self { known: HashMap::new(), counts: HashMap::new(), inflight: Default::default(), req, res }
    }

    /// Ask for a source's size if it is not known and not already asked for.
    fn want(&mut self, path: &Path) {
        if self.known.contains_key(path) || self.inflight.contains(path) {
            return;
        }
        self.inflight.insert(path.to_path_buf());
        let _ = self.req.send(path.to_path_buf());
    }

    /// Collect what the worker has read. True if anything landed, so the
    /// caller can repaint the listing.
    fn pump(&mut self) -> bool {
        let mut any = false;
        while let Ok((path, dims, frames)) = self.res.try_recv() {
            self.inflight.remove(&path);
            if let Some(n) = frames {
                self.counts.insert(path.clone(), n);
            }
            self.known.insert(path, dims);
            any = true;
        }
        any
    }
}

/// The width floor's verdict on one source: `known` is what the header
/// said, if it has been read. Unknown and unreadable both pass — the floor
/// hides what is too small, never what it cannot measure.
fn width_ok(min_width: u32, known: Option<&Option<(u32, u32)>>) -> bool {
    match known {
        Some(Some((w, _))) => *w >= min_width,
        _ => true,
    }
}


/// Tile sizes the size control and the zoom gestures stay within.
pub const TILE_MIN: f32 = imfram::gallery::MIN_SIZE;
pub const TILE_MAX: f32 = 640.0;

/// One thing in the view, in the order shown: a folder, a clip (a row of
/// the flat listing or a hit of the recursive search, with its versions
/// when they collapse), or a loose file nothing here decodes.
#[derive(Clone)]
enum Item {
    Dir(PathBuf),
    Seq { at: At, group: Option<VersionGroup> },
    Other(PathBuf),
}

/// Where a clip lives: a row of the flat listing, or a hit of the search.
#[derive(Clone, Copy, PartialEq, Debug)]
enum At {
    Flat(usize),
    Hit(usize),
}

/// An item's id, which the selection remembers: stable across sorts and
/// filters, and the same for a clip whichever view shows it.
fn hid(tag: u8, p: &Path) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    tag.hash(&mut h);
    p.hash(&mut h);
    h.finish()
}

fn seq_id(row: &SeqRow) -> u64 {
    hid(1, &row.frames[0])
}

/// What the thumbnail workers are handed for a clip: its frames, so frame
/// `n` of a scrub names a file (a sequence) or a frame inside one (a movie).
/// Equal by the first frame, which is what identifies the clip.
#[derive(Clone)]
pub struct Src {
    first: PathBuf,
    frames: Arc<[PathBuf]>,
    movie: bool,
}

impl PartialEq for Src {
    fn eq(&self, o: &Self) -> bool {
        self.first == o.first && self.movie == o.movie
    }
}
impl Eq for Src {}
impl Hash for Src {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.first.hash(h);
        self.movie.hash(h);
    }
}

impl Src {
    fn of(row: &SeqRow) -> Self {
        Self { first: row.frames[0].clone(), frames: row.frames.clone(), movie: row.movie }
    }
    /// For an item with no picture (never loaded).
    fn none(p: &Path) -> Self {
        Self { first: p.to_path_buf(), frames: Arc::from(Vec::new()), movie: false }
    }
    fn key(&self, f: u32) -> FrameKey {
        if self.movie {
            (self.first.clone(), f as u64)
        } else {
            (self.frames[(f as usize).min(self.frames.len().saturating_sub(1))].clone(), 0)
        }
    }
}

/// The gallery's thumbnail cache, decoding through the comp's own decoder
/// (EXR, DPX, stills, movies) with the viewer's tone curve.
fn tile_thumbs(decoder: Arc<dyn ExrDecoder>) -> imfram::gallery::Thumbs<Src> {
    imfram::gallery::Thumbs::new(THUMB_WORKERS, move |r: &Request<Src>| {
        if r.src.frames.is_empty() {
            return Err("no picture".into());
        }
        let max = r.max_px as usize;
        decode_thumb(&*decoder, &r.src.key(r.frame), |w, h| {
            let s = (max as f32 / w.max(h) as f32).min(1.0);
            (((w as f32 * s).round() as usize).max(1), ((h as f32 * s).round() as usize).max(1))
        })
        .ok_or_else(|| "could not decode".into())
    })
}

/// The view's items as the gallery and the list read them.
struct Rows<'a> {
    view: &'a [Item],
    listing: Option<&'a Listing>,
    search: Option<&'a Search>,
    root: &'a Path,
    binned: &'a HashSet<String>,
    counts: &'a HashMap<PathBuf, u64>,
}

/// A clip's folder and row.
fn lookup<'a>(listing: Option<&'a Listing>, search: Option<&'a Search>, root: &'a Path, at: At) -> Option<(&'a Path, &'a SeqRow)> {
    match at {
        At::Flat(i) => listing?.seqs.get(i).map(|r| (root, r)),
        At::Hit(i) => search?.hits.get(i).map(|(d, r)| (d.as_path(), r)),
    }
}

/// A search hit's folder, relative to where the search started.
fn rel_dir(root: &Path, dir: &Path) -> String {
    let rel = dir.strip_prefix(root).unwrap_or(dir).display().to_string();
    if rel.is_empty() {
        ".".into()
    } else {
        rel
    }
}

fn leaf(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
}

impl Rows<'_> {
    fn seq(&self, i: usize) -> Option<(&Path, &SeqRow, Option<&VersionGroup>)> {
        match &self.view[i] {
            Item::Seq { at, group } => lookup(self.listing, self.search, self.root, *at).map(|(d, r)| (d, r, group.as_ref())),
            _ => None,
        }
    }
    fn name(&self, i: usize) -> String {
        match &self.view[i] {
            Item::Dir(p) | Item::Other(p) => leaf(p),
            Item::Seq { .. } => self.seq(i).map(|(_, r, _)| r.pattern.clone()).unwrap_or_default(),
        }
    }
    fn binned(&self, dir: &Path, row: &SeqRow) -> bool {
        self.binned.contains(&dir.join(&row.pattern).display().to_string())
    }
}

impl Items for Rows<'_> {
    type Src = Src;
    fn len(&self) -> usize {
        self.view.len()
    }
    fn id(&self, i: usize) -> u64 {
        match &self.view[i] {
            Item::Dir(p) => hid(0, p),
            Item::Other(p) => hid(2, p),
            Item::Seq { .. } => self.seq(i).map(|(_, r, _)| seq_id(r)).unwrap_or(0),
        }
    }
    fn src(&self, i: usize) -> Src {
        match (&self.view[i], self.seq(i)) {
            (_, Some((_, r, _))) => Src::of(r),
            (Item::Dir(p) | Item::Other(p), None) => Src::none(p),
            _ => Src::none(Path::new("")),
        }
    }
    fn label(&self, i: usize) -> String {
        match (&self.view[i], self.seq(i)) {
            (Item::Seq { at: At::Hit(_), .. }, Some((d, r, _))) => format!("{}  ·  {}", r.pattern, rel_dir(self.root, d)),
            _ => self.name(i),
        }
    }
    fn frames(&self, i: usize) -> u32 {
        match self.seq(i) {
            Some((_, r, _)) if r.readable => {
                let n = if r.movie { self.counts.get(&r.frames[0]).copied().unwrap_or(1) as usize } else { r.frames.len() };
                n.clamp(1, u32::MAX as usize) as u32
            }
            _ => 1,
        }
    }
    fn frame_name(&self, i: usize, f: u32) -> String {
        match self.seq(i) {
            Some((_, r, _)) if !r.movie => r.numbers.get(f as usize).map(|n| n.to_string()).unwrap_or_default(),
            _ => (f + 1).to_string(),
        }
    }
    fn badge(&self, i: usize) -> Option<String> {
        let (d, r, g) = self.seq(i)?;
        let mut parts = vec![];
        if r.frames.len() > 1 {
            parts.push(format!("{}-{}", r.first, r.last));
        }
        if r.missing > 0 {
            parts.push(format!("⚠{}", r.missing));
        }
        if let Some(g) = g {
            parts.push(format!("v{:03} of {}", g.shown, g.versions.len()));
        }
        if self.binned(d, r) {
            parts.push("in bin".into());
        }
        (!parts.is_empty()).then(|| parts.join(" · "))
    }
    fn kind(&self, i: usize) -> Option<String> {
        match &self.view[i] {
            Item::Dir(_) => Some("folder".into()),
            Item::Other(p) => Some(ext_of(p).map(|e| e.to_uppercase()).unwrap_or_else(|| "file".into())),
            Item::Seq { .. } => self.seq(i).and_then(|(_, r, _)| r.ext.as_ref()).map(|e| e.to_uppercase()),
        }
    }
    fn has_picture(&self, i: usize) -> bool {
        self.seq(i).is_some_and(|(_, r, _)| r.readable)
    }
    fn dim(&self, i: usize) -> bool {
        match &self.view[i] {
            Item::Other(_) => true,
            Item::Seq { .. } => !self.has_picture(i),
            Item::Dir(_) => false,
        }
    }
}

/// The list's columns past the name. Folder only while searching.
fn columns(search: bool) -> Vec<Column> {
    let mut c = vec![
        Column::new("Name", 280.0),
        Column::new("Frames", 150.0).unsortable(),
        Column::new("Status", 150.0).unsortable(),
        Column::new("Duration", 76.0).align(egui::Align::RIGHT),
        Column::new("Modified", 96.0),
    ];
    if search {
        c.push(Column::new("Folder", 180.0).unsortable());
    }
    c
}

fn sort_col(k: SortKey) -> usize {
    match k {
        SortKey::Name => 0,
        SortKey::Duration => 3,
        SortKey::Date => 4,
    }
}

fn col_sort(c: usize) -> Option<SortKey> {
    match c {
        0 => Some(SortKey::Name),
        3 => Some(SortKey::Duration),
        4 => Some(SortKey::Date),
        _ => None,
    }
}

impl Details for Rows<'_> {
    fn cell(&self, i: usize, col: usize) -> String {
        let Some((d, r, g)) = self.seq(i) else {
            return match (&self.view[i], col) {
                (Item::Dir(_), 1) => "folder".into(),
                (Item::Other(_), 2) => "not decodable".into(),
                _ => String::new(),
            };
        };
        match col {
            1 if r.frames.len() == 1 => "single".into(),
            1 => format!("{}–{} ({})", r.first, r.last, r.frames.len()),
            2 => {
                let mut v = vec![];
                if r.missing > 0 {
                    v.push(format!("{} missing", r.missing));
                }
                if let Some(g) = g {
                    let latest = g.versions.last().copied().unwrap_or(g.shown);
                    v.push(if g.shown == latest { format!("v{:03} latest", g.shown) } else { format!("v{:03} of v{latest:03}", g.shown) });
                }
                if self.binned(d, r) {
                    v.push("in bin".into());
                }
                if !r.readable {
                    v.push("not decodable".into());
                }
                v.join(" · ")
            }
            3 => fmt_frames(r.duration()),
            4 => fmt_date(r.modified),
            5 => rel_dir(self.root, d),
            _ => String::new(),
        }
    }
}

pub struct SeqBrowser {
    /// Read node this browser is picking for; None = opened with L, which
    /// collects into the bin and spawns fresh Reads.
    pub target_node: Option<u64>,
    /// Picking a FOLDER for `target_node` (a Render's location), not a
    /// sequence: the action bar offers the directory on show.
    pub pick_folder: bool,
    /// The folder of the saved comp (`.wbl`), offered under Home; None
    /// while the comp is unsaved. The shell refreshes it every show.
    pub comp_dir: Option<PathBuf>,
    dir: PathBuf,
    filter: String,
    listing: Option<Result<Listing, String>>,
    /// Bumped on every scan, so the view knows to rebuild.
    listing_gen: u64,
    probes: HashMap<PathBuf, Probe>,
    decoder: Arc<dyn ExrDecoder>,
    // The inspector's picture: latest-wins single worker.
    thumb_tx: Sender<(FrameKey, egui::ColorImage)>,
    thumb_rx: Receiver<(FrameKey, egui::ColorImage)>,
    thumb_tex: Option<(FrameKey, egui::TextureHandle)>,
    thumb_inflight: Option<FrameKey>,
    thumb_wanted: Option<FrameKey>,
    /// Scrub position over the inspected clip, 0..1.
    scrub: f32,
    /// The preview transport, when it is running: where playback started
    /// from and when. Position is DERIVED from wall time rather than
    /// counted per repaint, so the preview runs at its stated rate whatever
    /// the UI's frame rate is doing — and a slow decode drops frames
    /// instead of playing the clip in slow motion.
    preview: Option<Preview>,
    /// Distinct frames the inspector thumbnail has actually SHOWN. Only a
    /// test reads it; it is what tells a moving preview from a stuck one.
    seen_frames: HashSet<FrameKey>,
    /// Tiles (imfram's gallery) instead of rows (its list).
    grid: bool,
    gallery: Gallery,
    list: List,
    /// Tile and row thumbnails, shared by both views.
    thumbs: imfram::gallery::Thumbs<Src>,
    /// Which file types the listing shows.
    types: TypeFilter,
    /// Collected sequences awaiting "create reads".
    bin: Vec<Picked>,
    /// Recursive search state (None = plain flat listing).
    search: Option<Search>,
    recursive: bool,
    depth: usize,
    /// Active sort, applied to the listing and to search hits alike.
    sort: SortKey,
    sort_desc: bool,
    /// Bumped whenever the hits are re-sorted in place.
    hits_sorted: u64,
    /// Show one entry per shot, standing for all its `v###` versions.
    collapse: bool,
    /// The version picked on a collapsed entry, by group key (absent =
    /// the latest, which is also what a new version landing shows).
    version_pick: HashMap<String, u32>,
    /// After a version pick: select the entry of this group once the view
    /// has been rebuilt round it.
    follow_group: Option<String>,
    /// Hide anything shorter than this many frames (0 = show everything).
    min_frames: usize,
    /// Hide anything narrower than this many pixels (0 = show everything).
    min_width: u32,
    /// Header-read sizes, for the width floor, and movies' frame counts.
    dims: Dims,
    /// Search hits need re-sorting (new hits landed, or the key changed).
    resort: bool,
    /// Extensions unticked in the types sidebar (lower-case; "" = none).
    excluded: BTreeSet<String>,
    /// The items in view order, and the inputs they were built from.
    view: Vec<Item>,
    view_key: u64,
    /// The selection, by item id; shared by both views.
    selection: BTreeSet<u64>,
    /// The one selected clip the inspector shows, and where it is.
    inspected: Option<(u64, At, Option<VersionGroup>)>,
    /// The item under the pointer last frame: Space's target.
    hovered: Option<u64>,
    path_bar: PathBar,
    tree: FolderTree,
    typeahead: TypeAhead,
    /// The right-click menu: on which item (None: empty space), and where.
    menu: Option<(Option<u64>, egui::Pos2)>,
    /// The big viewer, while it is up.
    quicklook: Option<QuickLook>,
    /// Frames for the big viewer.
    big: Thumbs,
}

impl SeqBrowser {
    pub fn new(target_node: Option<u64>, start: PathBuf, decoder: Arc<dyn ExrDecoder>) -> Self {
        let (thumb_tx, thumb_rx) = std::sync::mpsc::channel();
        let start = drive_root(&start);
        let mut gallery = Gallery::new();
        gallery.size = 160.0;
        gallery.metrics.aspect = 16.0 / 9.0;
        let mut list = List::new(columns(false));
        list.name_badge = false;
        list.sort = Some((sort_col(SortKey::Name), true));
        Self {
            target_node,
            pick_folder: false,
            dir: start,
            filter: String::new(),
            listing: None,
            listing_gen: 0,
            probes: HashMap::new(),
            decoder: decoder.clone(),
            thumb_tx,
            thumb_rx,
            thumb_tex: None,
            thumb_inflight: None,
            thumb_wanted: None,
            scrub: 0.0,
            preview: None,
            seen_frames: Default::default(),
            grid: false,
            gallery,
            list,
            thumbs: tile_thumbs(decoder.clone()),
            types: TypeFilter::Supported,
            bin: Vec::new(),
            search: None,
            recursive: false,
            depth: 10,
            sort: SortKey::Name,
            sort_desc: false,
            hits_sorted: 0,
            collapse: true,
            version_pick: HashMap::new(),
            follow_group: None,
            min_frames: 0,
            min_width: 0,
            dims: Dims::new(decoder.clone()),
            resort: false,
            excluded: BTreeSet::new(),
            view: Vec::new(),
            view_key: u64::MAX,
            selection: BTreeSet::new(),
            inspected: None,
            hovered: None,
            path_bar: PathBar::new(),
            tree: FolderTree::new(),
            comp_dir: None,
            typeahead: TypeAhead::new(),
            menu: None,
            quicklook: None,
            big: Thumbs::new(BIG_W, BIG_CAP),
        }
    }

    /// Everything the view is built from, hashed.
    fn view_key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.listing_gen.hash(&mut h);
        self.search.as_ref().map(|s| (s.generation, s.hits.len())).hash(&mut h);
        self.hits_sorted.hash(&mut h);
        self.filter.hash(&mut h);
        matches!(self.types, TypeFilter::All).hash(&mut h);
        self.excluded.hash(&mut h);
        self.min_frames.hash(&mut h);
        self.min_width.hash(&mut h);
        if self.min_width > 0 {
            self.dims.known.len().hash(&mut h);
        }
        (sort_col(self.sort), self.sort_desc, self.collapse).hash(&mut h);
        let mut picks: Vec<_> = self.version_pick.iter().collect();
        picks.sort();
        picks.hash(&mut h);
        h.finish()
    }

    /// The items in the order shown: folders (by name), then clips by the
    /// active sort, then loose files nothing decodes (when asked for).
    /// Search hits keep the order `resort` gave them.
    fn build_view(&self) -> Vec<Item> {
        let filter = self.filter.to_lowercase();
        let hit = |s: &str| filter.is_empty() || s.to_lowercase().contains(&filter);
        let mut out = Vec::new();
        if let Some(se) = &self.search {
            let rows: Vec<usize> = (0..se.hits.len())
                .filter(|&hi| {
                    let (d, r) = &se.hits[hi];
                    (hit(&r.pattern) || hit(&rel_dir(&self.dir, d))) && self.keep(r)
                })
                .collect();
            // Versions group across the tree (a `v003/` folder per version
            // is common), so the key is the whole path.
            let (hidden, groups) = if self.collapse {
                collapse_versions(rows.iter().map(|&hi| (hi, se.hits[hi].0.join(&se.hits[hi].1.pattern).display().to_string())), &self.version_pick)
            } else {
                Default::default()
            };
            for hi in rows.into_iter().filter(|hi| !hidden.contains(hi)) {
                out.push(Item::Seq { at: At::Hit(hi), group: groups.get(&hi).cloned() });
            }
            return out;
        }
        let Some(Ok(l)) = &self.listing else { return out };
        let mut dirs: Vec<&PathBuf> = l.dirs.iter().filter(|d| l.dir_on(d, self.collapse) && hit(&leaf(d))).collect();
        dirs.sort_by(|a, b| natural_cmp(&leaf(a), &leaf(b)));
        out.extend(dirs.into_iter().map(|d| Item::Dir(d.clone())));
        let mut rows: Vec<(usize, &SeqRow)> = l.seqs.iter().enumerate().filter(|(i, s)| l.row_on(*i, self.collapse) && hit(&s.pattern) && self.keep(s)).collect();
        let (hidden, groups) = if self.collapse { collapse_versions(rows.iter().map(|(i, s)| (*i, s.pattern.clone())), &self.version_pick) } else { Default::default() };
        rows.retain(|(i, _)| !hidden.contains(i));
        let (key, desc) = (self.sort, self.sort_desc);
        rows.sort_by(|(_, a), (_, b)| row_cmp(a, b, key, desc));
        out.extend(rows.into_iter().map(|(i, _)| Item::Seq { at: At::Flat(i), group: groups.get(&i).cloned() }));
        if self.min_frames <= 1 {
            out.extend(l.other.iter().filter(|f| hit(&leaf(f)) && self.shows(ext_of(f).as_deref())).map(|f| Item::Other(f.clone())));
        }
        out
    }

    fn rows<'a>(&'a self, binned: &'a HashSet<String>) -> Rows<'a> {
        Rows {
            view: &self.view,
            listing: self.listing.as_ref().and_then(|l| l.as_ref().ok()),
            search: self.search.as_ref(),
            root: &self.dir,
            binned,
            counts: &self.dims.counts,
        }
    }

    fn binned_set(&self) -> HashSet<String> {
        self.bin.iter().map(|b| b.display.clone()).collect()
    }

    /// The view's index of the item with this id.
    fn index_of(&self, id: u64) -> Option<usize> {
        let binned = HashSet::new();
        let rows = self.rows(&binned);
        (0..rows.len()).find(|&i| rows.id(i) == id)
    }

    /// A clip by id: its folder, row and versions.
    fn seq_by_id(&self, id: u64) -> Option<(PathBuf, SeqRow, Option<VersionGroup>)> {
        let i = self.index_of(id)?;
        let binned = HashSet::new();
        self.rows(&binned).seq(i).map(|(d, r, g)| (d.to_path_buf(), r.clone(), g.cloned()))
    }

    /// The selected clips that can load, with their folders, in view order.
    fn selected_seqs(&self) -> Vec<(PathBuf, SeqRow)> {
        let binned = HashSet::new();
        let rows = self.rows(&binned);
        (0..rows.len())
            .filter(|&i| self.selection.contains(&rows.id(i)))
            .filter_map(|i| rows.seq(i))
            .filter(|(_, r, _)| r.readable)
            .map(|(d, r, _)| (d.to_path_buf(), r.clone()))
            .collect()
    }

    /// Re-sort the hits if asked, and rebuild the view when anything it is
    /// made of changed; a selection the view no longer shows is dropped.
    fn refresh_view(&mut self) {
        if self.resort {
            if let Some(se) = self.search.as_mut() {
                let (key, desc) = (self.sort, self.sort_desc);
                se.hits.sort_by(|(_, a), (_, b)| row_cmp(a, b, key, desc));
            }
            self.resort = false;
            self.hits_sorted += 1;
        }
        let k = self.view_key();
        if k == self.view_key {
            return;
        }
        self.view = self.build_view();
        self.view_key = k;
        let binned = HashSet::new();
        let rows = self.rows(&binned);
        let ids: HashSet<u64> = (0..rows.len()).map(|i| rows.id(i)).collect();
        let follow = self.follow_group.as_ref().and_then(|g| (0..rows.len()).find(|&i| matches!(&self.view[i], Item::Seq { group: Some(vg), .. } if vg.key == *g)).map(|i| rows.id(i)));
        self.follow_group = None;
        if let Some(id) = follow {
            self.selection = [id].into();
        }
        self.selection.retain(|id| ids.contains(id));
        self.inspected = None;
    }

    /// The inspector follows the selection: one selected clip is inspected
    /// (a new one starts at its first frame, transport stopped).
    fn sync_inspected(&mut self) {
        let one = (self.selection.len() == 1).then(|| *self.selection.iter().next().unwrap());
        if self.inspected.as_ref().map(|i| i.0) == one && one.is_some() {
            return;
        }
        let was = self.inspected.as_ref().map(|i| i.0);
        self.inspected = one.and_then(|id| {
            let i = self.index_of(id)?;
            match &self.view[i] {
                Item::Seq { at, group } => Some((id, *at, group.clone())),
                _ => None,
            }
        });
        let now = self.inspected.as_ref().map(|i| i.0);
        if now != was {
            self.scrub = 0.0;
            self.stop_preview();
            if let Some((_, at, _)) = &self.inspected {
                let key = lookup(self.listing.as_ref().and_then(|l| l.as_ref().ok()), self.search.as_ref(), &self.dir, *at).map(|(_, r)| r.frame_key(0));
                if let Some(k) = key {
                    self.request_thumb(k);
                }
            }
        }
    }

    /// --test-thumbs: switch view/filter and read back what the listing
    /// would show, plus how many thumbnails have actually decoded.
    pub fn test_set_grid(&mut self, on: bool) {
        self.grid = on;
    }

    pub fn test_set_tile(&mut self, size: f32) {
        self.gallery.size = size.clamp(TILE_MIN, TILE_MAX);
    }

    pub fn test_tile(&self) -> f32 {
        self.gallery.size
    }

    pub fn test_set_filter(&mut self, f: TypeFilter) {
        self.types = f;
    }

    /// (sequence rows passing the filter, loose files passing it, thumbs
    /// decoded so far)
    pub fn test_counts(&self) -> (usize, usize, usize) {
        let (seqs, other) = self
            .listing
            .as_ref()
            .and_then(|l| l.as_ref().ok())
            .map(|l| (l.seqs.iter().filter(|s| self.keep(s)).count(), l.other.iter().filter(|f| self.shows(ext_of(f).as_deref())).count()))
            .unwrap_or((0, 0));
        (seqs, other, self.thumbs.stats().textures)
    }

    /// --test-bin: drive a recursive search and collect its hits.
    pub fn test_search(&mut self, depth: usize) {
        self.depth = depth;
        self.recursive = true;
        self.start_search();
    }

    /// (sequences found, directories walked, still running)
    pub fn test_search_status(&self) -> (usize, usize, bool) {
        self.search.as_ref().map(|s| (s.hits.len(), s.scanned, s.running)).unwrap_or((0, 0, false))
    }

    /// --test-scrub: select row `i`, then drive the transport directly.
    /// Returns (frames scrubbable, whether the preview is running).
    pub fn test_select_and_len(&mut self, i: usize) -> (usize, bool) {
        self.test_select(i);
        let probed = self.listing.as_ref().and_then(|l| l.as_ref().ok()).and_then(|l| l.seqs.get(i)).cloned();
        match probed {
            Some(row) => {
                let n = self.probe(&row.frames[0].clone(), &row.frames).frames;
                (row.scrub_len(n), self.preview.is_some())
            }
            None => (0, false),
        }
    }

    pub fn test_play(&mut self, len: usize) {
        self.start_preview(len);
    }

    /// The transport's current frame, and whether it is running.
    pub fn test_preview_state(&mut self, len: usize) -> (usize, bool) {
        self.pump_preview();
        (self.scrub_index(len), self.preview.is_some())
    }

    /// How many DISTINCT frames the inspector has shown — the thing that
    /// separates a running player from a stuck one.
    pub fn test_thumb_frames_seen(&self) -> usize {
        self.seen_frames.len()
    }

    /// Bin the first `n` hits and hand back the batch, as the bin button does.
    pub fn test_bin_hits(&mut self, n: usize) -> BrowserAction {
        let picks: Vec<(PathBuf, SeqRow)> = self.search.as_ref().map(|s| s.hits.iter().take(n).cloned().collect()).unwrap_or_default();
        for (d, r) in &picks {
            self.add_to_bin(d, r);
        }
        BrowserAction::PickMany(std::mem::take(&mut self.bin))
    }

    /// Test hooks (--test-browser): (scanned seq rows, thumb texture ready).
    pub fn test_status(&self) -> (usize, bool) {
        let n = self.listing.as_ref().and_then(|l| l.as_ref().ok()).map(|l| l.seqs.len()).unwrap_or(0);
        (n, self.thumb_tex.is_some())
    }

    /// --test-browser: drive the sort as a header click would.
    pub fn test_set_sort(&mut self, sort: SortKey, desc: bool) {
        self.sort = sort;
        self.sort_desc = desc;
        self.resort = true;
    }

    pub fn test_set_min_frames(&mut self, n: usize) {
        self.min_frames = n;
    }

    #[cfg(test)]
    pub fn test_set_min_width(&mut self, w: u32) {
        self.min_width = w;
    }

    #[cfg(test)]
    /// Sizes still being read for the width floor.
    pub fn test_dims_pending(&self) -> usize {
        self.dims.inflight.len()
    }

    /// (pattern, duration) for the clips the flat listing shows, in
    /// display order — the view the gallery and the list draw.
    pub fn test_visible(&self) -> Vec<(String, usize)> {
        let listing = self.listing.as_ref().and_then(|l| l.as_ref().ok());
        self.build_view()
            .iter()
            .filter_map(|it| match it {
                Item::Seq { at, .. } => lookup(listing, None, &self.dir, *at),
                _ => None,
            })
            .map(|(_, r)| (r.pattern.clone(), r.duration()))
            .collect()
    }

    /// Select row `i` of the flat listing as a click would (kicks off probe
    /// + thumbnail).
    pub fn test_select(&mut self, i: usize) {
        let id = self.listing.as_ref().and_then(|l| l.as_ref().ok()).and_then(|l| l.seqs.get(i)).map(seq_id);
        if let Some(id) = id {
            self.refresh_view();
            self.selection = [id].into();
            self.inspected = None;
            self.sync_inspected();
        }
    }

    /// Does this row survive the type gate, the duration floor and the
    /// width floor?
    fn keep(&self, row: &SeqRow) -> bool {
        self.shows(row.ext.as_deref()) && row.duration() >= self.min_frames && (self.min_width == 0 || width_ok(self.min_width, self.dims.known.get(&row.frames[0])))
    }

    /// With a width floor set, ask for the size of every readable row that
    /// has not been measured yet. O(rows) hash probes per frame; the reads
    /// themselves happen on the `Dims` thread.
    fn request_dims(&mut self) {
        if self.min_width == 0 {
            return;
        }
        let mut paths: Vec<PathBuf> = Vec::new();
        if let Some(se) = &self.search {
            paths.extend(se.hits.iter().filter(|(_, r)| r.readable).map(|(_, r)| r.frames[0].clone()));
        } else if let Some(l) = self.listing.as_ref().and_then(|l| l.as_ref().ok()) {
            paths.extend(l.seqs.iter().filter(|r| r.readable).map(|r| r.frames[0].clone()));
        }
        for p in paths {
            self.dims.want(&p);
        }
    }

    fn add_to_bin(&mut self, dir: &Path, row: &SeqRow) {
        let p = pick_of(dir, row);
        if !self.bin.iter().any(|b| b.display == p.display) {
            self.bin.push(p);
        }
    }

    /// Every selected, readable clip into the bin (once each — the bin
    /// dedupes). The selection is kept, so the "in bin" badges show what
    /// went in.
    fn bin_selection(&mut self) {
        for (d, r) in self.selected_seqs() {
            self.add_to_bin(&d, &r);
        }
    }

    /// Clips dragged onto the bin, by id.
    fn bin_ids(&mut self, ids: &[u64]) {
        let picks: Vec<(PathBuf, SeqRow)> = ids.iter().filter_map(|&id| self.seq_by_id(id)).filter(|(_, r, _)| r.readable).map(|(d, r, _)| (d, r)).collect();
        for (d, r) in &picks {
            self.add_to_bin(d, r);
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, pinned: &mut Vec<PathBuf>, recent: &[PathBuf]) -> BrowserAction {
        if self.listing.is_none() {
            self.scan();
        }
        self.pump_thumbs(ctx);
        self.big.pump(ctx, &self.decoder);
        self.pump_search();
        if self.dims.pump() {
            ctx.request_repaint();
        }
        self.request_dims();
        if self.pump_preview() {
            // The browser is an idle window; without asking, the transport
            // would only advance when something else caused a repaint.
            ctx.request_repaint();
        }
        self.refresh_view();
        self.sync_inspected();

        let mut action = BrowserAction::None;
        let mut open = true;
        let title = match self.target_node {
            _ if self.pick_folder => "render location — pick a folder",
            Some(_) => "sequences",
            None => "sequences — collect into bin",
        };
        let mut preview_ask: Option<Option<u64>> = None;
        // Full screen by default: this is a picking errand, not a palette,
        // and thumbnails want the room. It stays a window, so it can still
        // be dragged smaller to see the graph behind it.
        egui::Window::new(title).open(&mut open).collapsible(false).default_rect(ctx.content_rect()).show(ctx, |ui| {
            let mut nav: Option<PathBuf> = None;

            // --- path bar, then the view's controls ---
            let fav = pinned.contains(&self.dir);
            let pe = self.path_bar.show(ui, &self.dir, Some(fav));
            if let Some(g) = pe.go {
                nav = Some(g);
            }
            if pe.rescan {
                self.rescan();
            }
            match pe.favourite {
                Some(true) => pinned.push(self.dir.clone()),
                Some(false) => pinned.retain(|p| *p != self.dir),
                None => {}
            }
            self.toolbar(ui);
            ui.separator();

            // --- left: favourites, recent, the folder tree, the bin ---
            egui::Panel::left(egui::Id::new("seq-side")).resizable(true).default_size(210.0).show(ui, |ui| {
                egui::Panel::bottom(egui::Id::new("seq-bin")).resizable(false).show(ui, |ui| self.bin_ui(ui));
                egui::ScrollArea::vertical().id_salt("seq-side-scroll").auto_shrink([false, false]).show(ui, |ui| {
                    if let Some(h) = nav::home_place(ui, &self.dir) {
                        nav = Some(h);
                    }
                    if let Some(c) = &self.comp_dir {
                        ui.add_space(-6.0);
                        if ui.selectable_label(*c == self.dir, "Comp folder").on_hover_text(format!("where the saved comp lives\n{}", c.display())).clicked() {
                            nav = Some(c.clone());
                        }
                        ui.add_space(6.0);
                    }
                    let ev = nav::places(ui, "Favourites", pinned, &self.dir, true, Some("☆ in the path bar adds the folder shown"));
                    if let Some(p) = ev.go {
                        nav = Some(p);
                    }
                    if let Some(r) = ev.remove {
                        pinned.retain(|p| *p != r);
                    }
                    if let Some(p) = nav::places(ui, "Recent", recent, &self.dir, false, None).go {
                        nav = Some(p);
                    }
                    ui.strong("Folders");
                    if let Some(p) = self.tree.show(ui, &self.dir) {
                        nav = Some(p);
                    }
                });
            });

            // --- right: what is here, and what to hide ---
            egui::Panel::right(egui::Id::new("seq-types")).resizable(true).default_size(210.0).show(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("seq-types-scroll").show(ui, |ui| self.types_sidebar(ui));
            });

            // --- action bar: the commit button, bottom right, where a
            // dialog's confirm belongs. Outermost bottom panel, so it
            // sits below the inspector rather than inside it.
            let n_sel = self.selected_seqs().len();
            if self.pick_folder {
                egui::Panel::bottom(egui::Id::new("seq-actions")).resizable(false).show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.weak("frames go in the folder shown; a missing one is made at render");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let label = RichText::new("use this folder").strong().color(theme::text_hot()).size(14.0);
                            if ui
                                .add(egui::Button::new(label).fill(theme::browser_accent()).corner_radius(6.0).min_size(egui::vec2(170.0, 30.0)))
                                .on_hover_text(self.dir.display().to_string())
                                .clicked()
                            {
                                action = BrowserAction::PickDir(self.dir.clone());
                            }
                        });
                    });
                    ui.add_space(4.0);
                });
            } else if !self.bin.is_empty() || n_sel > 0 {
                egui::Panel::bottom(egui::Id::new("seq-actions")).resizable(false).show(ui, |ui| {
                    let acts = action_row(ui, n_sel, self.bin.len());
                    if acts.add {
                        self.bin_selection();
                    }
                    if acts.commit {
                        action = BrowserAction::PickMany(std::mem::take(&mut self.bin));
                    }
                });
            }

            // --- inspector (bottom) for the one selected clip ---
            let inspected = self.inspected.clone().and_then(|(_, at, g)| {
                lookup(self.listing.as_ref().and_then(|l| l.as_ref().ok()), self.search.as_ref(), &self.dir, at).map(|(d, r)| (d.to_path_buf(), r.clone(), g))
            });
            if let Some((dir, row, group)) = inspected {
                egui::Panel::bottom(egui::Id::new("seq-insp")).resizable(false).show(ui, |ui| {
                    self.inspector_ui(ui, &dir, &row, group.as_ref(), &mut action);
                });
            }

            // --- the items ---
            if let Some(p) = self.items_ui(ui, &mut action, &mut preview_ask) {
                nav = Some(p);
            }
            if let Some(p) = nav {
                self.navigate(p);
            }
        });
        // Space: the big viewer for the clip under the pointer (else the
        // selected one), and Space again puts it away. A key EVENT, not a
        // held state, so holding it does not flicker the viewer. (With the
        // view focused it takes Space itself and answers `preview`.)
        let space = !ctx.egui_wants_keyboard_input() && ctx.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Key { key: egui::Key::Space, pressed: true, repeat: false, .. })));
        if space {
            preview_ask = Some(None);
        }
        if let Some(target) = preview_ask {
            self.toggle_quicklook(target);
        }
        self.menu_ui(ctx, &mut action);
        self.quicklook_ui(ctx, &mut action);
        if self.inspected.is_none() {
            self.stop_preview();
        }
        if !open {
            self.stop_preview();
            self.quicklook = None;
            action = BrowserAction::Close;
        }
        action
    }

    /// Filter, search, versions; tiles or rows, their size and the sort.
    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("filter · Enter searches subfolders").desired_width(200.0).id(egui::Id::new("seq.filter")));
            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) && !self.recursive {
                self.recursive = true;
                self.start_search();
            }
            if r.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                if self.recursive {
                    self.exit_search();
                } else {
                    self.filter.clear();
                }
            }
            if ui.selectable_label(self.recursive, "recursive").on_hover_text("search this folder and everything under it").clicked() {
                if self.recursive {
                    self.exit_search();
                } else {
                    self.recursive = true;
                    self.start_search();
                }
            }
            // Recursive search: sequitur groups each directory, we walk
            // the tree and stream what it finds.
            if ui.add(egui::DragValue::new(&mut self.depth).update_while_editing(false).speed(crate::widgets::rated(ui, 1.0)).range(1..=12).prefix("depth ")).changed() && self.recursive {
                self.start_search();
            }
            if ui.selectable_label(self.collapse, "versions").on_hover_text("one entry per shot: v001, v002, v003 … show as the latest; right-click or the inspector picks an older one").clicked() {
                self.collapse = !self.collapse;
            }
            ui.separator();
            if ui.selectable_label(self.grid, "grid").on_hover_text("tiles: move across one to scrub it").clicked() {
                self.grid = true;
            }
            if ui.selectable_label(!self.grid, "list").on_hover_text("rows: sort by a column's header").clicked() {
                self.grid = false;
            }
            if self.grid {
                ui.add(egui::Slider::new(&mut self.gallery.size, TILE_MIN..=TILE_MAX).show_value(false)).on_hover_text("tile size (cmd-wheel, + / -)");
                // The list sorts from its column headers; tiles have none,
                // so the same state gets a control here.
                let mut picked = None;
                egui::ComboBox::from_id_salt("seq-sort").selected_text(self.sort.label()).width(96.0).show_ui(ui, |ui| {
                    for k in [SortKey::Name, SortKey::Date, SortKey::Duration] {
                        if ui.selectable_label(self.sort == k, k.label()).clicked() {
                            picked = Some(k);
                        }
                    }
                });
                if let Some(k) = picked {
                    self.sort = k;
                    self.sort_desc = k.desc_by_default();
                    self.resort = true;
                }
                if ui.button(if self.sort_desc { "⏷" } else { "⏶" }).on_hover_text("sort direction").clicked() {
                    self.sort_desc = !self.sort_desc;
                    self.resort = true;
                }
            } else {
                ui.add(egui::Slider::new(&mut self.list.row_h, imfram::gallery::list::MIN_ROW..=imfram::gallery::list::MAX_ROW).show_value(false)).on_hover_text("row size (cmd-wheel, + / -)");
            }
        });
    }

    /// Back to the folder from a search.
    fn exit_search(&mut self) {
        self.recursive = false;
        self.search = None;
        self.selection.clear();
    }

    /// Scan the folder (and the search, and the tree) again.
    fn rescan(&mut self) {
        self.listing = None;
        self.probes.clear();
        self.tree.refresh();
        if self.recursive {
            self.start_search();
        }
    }

    /// The bin: what is collected, × to take one out; clips dragged here
    /// (or B, or the action bar) go in.
    fn bin_ui(&mut self, ui: &mut egui::Ui) {
        let r = egui::Frame::group(ui.style())
            .show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.strong("Bin");
                    if !self.bin.is_empty() {
                        ui.weak(self.bin.len().to_string());
                    }
                });
                if self.bin.is_empty() {
                    ui.weak("drag clips here, or select and press B");
                    return;
                }
                let mut drop_i: Option<usize> = None;
                egui::ScrollArea::vertical().id_salt("seq-bin-scroll").max_height(160.0).show(ui, |ui| {
                    for (i, b) in self.bin.iter().enumerate() {
                        ui.horizontal(|ui| {
                            if ui.add(egui::Button::new("×").frame(false)).on_hover_text("remove").clicked() {
                                drop_i = Some(i);
                            }
                            ui.small(RichText::new(format!("{}  ({})", leaf(Path::new(&b.display)), b.frames.len())).weak()).on_hover_text(&b.display);
                        });
                    }
                });
                if let Some(i) = drop_i {
                    self.bin.remove(i);
                }
                if ui.small_button("empty bin").clicked() {
                    self.bin.clear();
                }
            })
            .response;
        if let Some(d) = r.dnd_release_payload::<Dragged>() {
            self.bin_ids(&d.0);
        }
        if r.dnd_hover_payload::<Dragged>().is_some() {
            ui.painter().rect_stroke(r.rect.expand(1.0), 4.0, egui::Stroke::new(2.0, ui.visuals().selection.bg_fill), egui::StrokeKind::Outside);
        }
    }

    /// What is in view, then the gallery or the list over it. Answers a
    /// folder to go into.
    fn items_ui(&mut self, ui: &mut egui::Ui, action: &mut BrowserAction, preview: &mut Option<Option<u64>>) -> Option<PathBuf> {
        self.count_line(ui);
        match &self.listing {
            #[cfg(target_arch = "wasm32")]
            Some(Err(_)) if self.search.is_none() => {
                web_hints(ui);
                return None;
            }
            #[cfg(not(target_arch = "wasm32"))]
            Some(Err(e)) if self.search.is_none() => {
                ui.colored_label(theme::error(), e);
                return None;
            }
            _ => {}
        }
        let want_cols = columns(self.search.is_some()).len();
        if self.list.columns.len() != want_cols {
            self.list.columns = columns(self.search.is_some());
        }
        let binned = self.binned_set();
        let rows = Rows {
            view: &self.view,
            listing: self.listing.as_ref().and_then(|l| l.as_ref().ok()),
            search: self.search.as_ref(),
            root: &self.dir,
            binned: &binned,
            counts: &self.dims.counts,
        };
        let before_sort = self.list.sort;
        let (ev, vis) = if self.grid {
            self.gallery.selection = self.selection.clone();
            let ev = self.gallery.show(ui, &rows, &mut self.thumbs);
            self.gallery.size = self.gallery.size.clamp(TILE_MIN, TILE_MAX);
            self.selection = self.gallery.selection.clone();
            self.hovered = self.gallery.hovered();
            (ev, self.gallery.visible())
        } else {
            self.list.sort = Some((sort_col(self.sort), !self.sort_desc));
            let before = self.list.sort;
            self.list.selection = self.selection.clone();
            let ev = self.list.show(ui, &rows, &mut self.thumbs);
            self.selection = self.list.selection.clone();
            self.hovered = self.list.hovered();
            if self.list.sort != before {
                match self.list.sort.and_then(|(c, up)| col_sort(c).map(|k| (k, up))) {
                    Some((k, up)) => {
                        self.sort = k;
                        self.sort_desc = !up;
                        self.resort = true;
                    }
                    None => self.list.sort = before_sort,
                }
            }
            (ev, self.list.visible())
        };
        // A movie's length is a header read: asked for the movies in view
        // only, on the Dims thread, so a scrub across one has its frames.
        let movies: Vec<PathBuf> = vis.filter(|&i| i < rows.len()).filter_map(|i| rows.seq(i)).filter(|(_, r, _)| r.movie && r.readable).map(|(_, r, _)| r.frames[0].clone()).collect();
        let names: Vec<String> = (0..rows.len()).map(|i| rows.name(i)).collect();
        let ids_of = |i: usize| rows.id(i);
        let jump = self.typeahead.feed(ui, "bB", names.iter().map(String::as_str)).map(ids_of);
        for m in movies {
            self.dims.want(&m);
        }
        if let Some(id) = jump {
            self.selection = [id].into();
            if let Some(i) = self.index_of(id) {
                self.gallery.scroll_to(i);
                self.list.scroll_to(i);
            }
        }
        // B: the selection into the bin.
        if !ui.ctx().egui_wants_keyboard_input() && ui.ui_contains_pointer() && ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::B)) {
            self.bin_selection();
        }
        self.events(ui, ev, action, preview)
    }

    fn events(&mut self, ui: &egui::Ui, ev: GalleryEvent, action: &mut BrowserAction, preview: &mut Option<Option<u64>>) -> Option<PathBuf> {
        // A plain click on a folder goes into it (as seq_manip, and Finder's
        // column view); a modified click only selects it.
        if ev.selection_changed && self.selection.len() == 1 && ui.input(|i| i.pointer.primary_released() && i.modifiers.is_none()) {
            let id = *self.selection.iter().next().unwrap();
            if let Some(Item::Dir(p)) = self.index_of(id).map(|i| self.view[i].clone()) {
                return Some(p);
            }
        }
        if let Some(id) = ev.activated {
            match self.index_of(id).map(|i| self.view[i].clone()) {
                Some(Item::Dir(p)) => return Some(p),
                Some(Item::Seq { .. }) => {
                    if let Some((d, r, _)) = self.seq_by_id(id).filter(|(_, r, _)| r.readable) {
                        *action = pick(&d, &r);
                    }
                }
                _ => {}
            }
        }
        if let Some(id) = ev.preview {
            *preview = Some(Some(id));
        }
        if let Some(c) = ev.context {
            self.menu = Some(c);
        }
        None
    }

    /// One line over the items: what is here and what is picked.
    fn count_line(&self, ui: &mut egui::Ui) {
        let binned = HashSet::new();
        let rows = self.rows(&binned);
        let clips = self.view.iter().filter(|i| matches!(i, Item::Seq { .. })).count();
        let dirs = self.view.iter().filter(|i| matches!(i, Item::Dir(_))).count();
        ui.horizontal(|ui| {
            if let Some(se) = &self.search {
                if se.running {
                    ui.spinner();
                }
                ui.weak(format!("{} clips in {} folders walked{} under {}", se.hits.len(), se.scanned, if se.running { " …" } else { "" }, self.dir.display()));
                if !self.filter.is_empty() {
                    ui.weak(format!("— {clips} match “{}”", self.filter));
                }
            } else {
                ui.weak(format!("{dirs} folders, {clips} clips"));
            }
            let sel = self.selection.len();
            if sel > 0 {
                ui.weak(format!("— {sel} selected"));
            }
            if !self.typeahead.typed().is_empty() {
                ui.weak(format!("— “{}”", self.typeahead.typed()));
            }
            let _ = rows;
        });
    }

    /// The right-click menu over the item it was opened on.
    fn menu_ui(&mut self, ctx: &egui::Context, action: &mut BrowserAction) {
        let Some((on, pos)) = self.menu else { return };
        let item = on.and_then(|id| self.index_of(id)).map(|i| self.view[i].clone());
        let seq = on.and_then(|id| self.seq_by_id(id));
        let n_sel = self.selected_seqs().len();
        let screen = ctx.content_rect();
        let pos = egui::pos2(pos.x.min(screen.right() - 200.0), pos.y.min(screen.bottom() - 260.0).max(screen.top()));
        let mut close = false;
        let mut quick = false;
        let area = egui::Area::new(egui::Id::new("seq-menu")).order(egui::Order::Foreground).fixed_pos(pos).show(ctx, |ui| {
            egui::Frame::menu(ui.style()).show(ui, |ui| {
                ui.set_min_width(180.0);
                let item_btn = |ui: &mut egui::Ui, on: bool, t: &str| ui.add_enabled(on, egui::Button::new(t).frame(false)).clicked();
                if let Some(Item::Dir(p)) = &item {
                    if item_btn(ui, true, "open") {
                        self.navigate(p.clone());
                        close = true;
                    }
                }
                let readable = seq.as_ref().is_some_and(|(_, r, _)| r.readable);
                if item_btn(ui, readable, "use in Load") {
                    if let Some((d, r, _)) = &seq {
                        *action = pick(d, r);
                    }
                    close = true;
                }
                if item_btn(ui, readable, "quick look  (space)") {
                    quick = true;
                    close = true;
                }
                if item_btn(ui, n_sel > 0, &format!("add {n_sel} to bin  (B)")) {
                    self.bin_selection();
                    close = true;
                }
                if let Some((_, _, Some(g))) = &seq {
                    ui.separator();
                    ui.weak(format!("{} versions", g.versions.len()));
                    let latest = g.versions.last().copied().unwrap_or(g.shown);
                    for &v in g.versions.iter().rev() {
                        let label = if v == latest { format!("v{v:03}  latest") } else { format!("v{v:03}") };
                        if ui.add(egui::Button::selectable(v == g.shown, label).frame(false)).clicked() {
                            self.version_pick.insert(g.key.clone(), v);
                            self.follow_group = Some(g.key.clone());
                            close = true;
                        }
                    }
                }
                ui.separator();
                let path = match (&item, &seq) {
                    (_, Some((d, r, _))) => Some(d.join(&r.pattern)),
                    (Some(Item::Dir(p) | Item::Other(p)), _) => Some(p.clone()),
                    _ => None,
                };
                if item_btn(ui, path.is_some(), "copy path") {
                    if let Some(p) = path {
                        ui.ctx().copy_text(p.display().to_string());
                    }
                    close = true;
                }
            });
        });
        if quick {
            self.toggle_quicklook(on);
        }
        let clicked_off = ctx.input(|i| i.pointer.any_pressed()) && !area.response.contains_pointer();
        if close || ctx.input(|i| i.key_pressed(egui::Key::Escape)) || clicked_off {
            self.menu = None;
        }
    }

    /// Space: open the quick look on `target` (else the item under the
    /// pointer, else the one selected clip), or put it away.
    fn toggle_quicklook(&mut self, target: Option<u64>) {
        if self.quicklook.is_some() {
            self.quicklook = None;
            return;
        }
        let one = (self.selection.len() == 1).then(|| *self.selection.iter().next().unwrap());
        let hovered = self.hovered.filter(|id| self.seq_by_id(*id).is_some());
        if let Some((d, r, _)) = target.or(hovered).or(one).and_then(|id| self.seq_by_id(id)) {
            self.open_quicklook(d, r);
        }
    }


    /// Kick off (or restart) a recursive scan from the current directory.
    fn start_search(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        let generation = self.search.as_ref().map(|s| s.generation + 1).unwrap_or(1);
        self.search = Some(Search {
            rx,
            generation,
            scanned: 0,
            running: true,
            hits: Vec::new(),
            other: BTreeMap::new(),
        });
        self.resort = true;
        self.selection.clear();
        let root = self.dir.clone();
        let depth = self.depth;
        // sequitur groups a directory at a time (its scan is deliberately
        // flat), so the walk is ours: breadth-first, bounded, streaming each
        // directory's sequences as they are found.
        crate::bg::spawn(move || {
            let mut queue = vec![(root, 0usize)];
            while let Some((dir, d)) = queue.pop() {
                let Ok(listing) = FileSequence::parse_directory(&dir, 2) else {
                    continue;
                };
                let mut rows = Vec::new();
                let mut other = Vec::new();
                for e in &listing.entries {
                    if e.hidden {
                        continue;
                    }
                    if matches!(e.kind, EntryKind::Directory) && d < depth {
                        queue.push((e.path.clone(), d + 1));
                    }
                    if matches!(e.kind, EntryKind::File) {
                        if engine::decode::is_supported(&e.path) {
                            rows.push(single_row(&e.path));
                        } else {
                            other.push(ext_key(ext_of(&e.path).as_deref()));
                        }
                    }
                }
                for sq in &listing.sequences {
                    rows.extend(rows_of(sq));
                }
                rows.sort_by(|a, b| natural_cmp(&a.pattern, &b.pattern));
                if tx.send((generation, dir, rows, other)).is_err() {
                    return; // browser closed
                }
            }
            // done marker
            let _ = tx.send((generation, PathBuf::new(), Vec::new(), Vec::new()));
        });
    }

    fn pump_search(&mut self) {
        let mut arrived = false;
        {
            let Some(se) = &mut self.search else { return };
            let gen = se.generation;
            let mut msgs = Vec::new();
            while let Ok(m) = se.rx.try_recv() {
                msgs.push(m);
            }
            for (g, dir, rows, other) in msgs {
                if g != gen {
                    continue;
                }
                if dir.as_os_str().is_empty() {
                    se.running = false;
                    continue;
                }
                se.scanned += 1;
                for r in rows {
                    arrived = true;
                    se.hits.push((dir.clone(), r));
                }
                for e in other {
                    *se.other.entry(e).or_default() += 1;
                }
            }
        }
        // Hits land at the tail; the render sorts them back into place.
        self.resort |= arrived;
    }

    fn navigate(&mut self, to: PathBuf) {
        let to = drive_root(&engine::paths::expand_home(&to.to_string_lossy()));
        self.search = None;
        self.recursive = false;
        self.selection.clear();
        self.inspected = None;
        self.menu = None;
        self.dir = to;
        self.listing = None;
        self.scrub = 0.0;
        self.gallery.scroll_to(0);
        self.list.scroll_to(0);
    }

    fn scan(&mut self) {
        let result = FileSequence::parse_directory(&self.dir, 2)
            .map_err(|e| format!("{e}"))
            .map(|l| {
                let mut dirs = Vec::new();
                let mut other = Vec::new();
                let mut seqs: Vec<SeqRow> = Vec::new();
                for e in l.entries {
                    if e.hidden {
                        continue;
                    }
                    match e.kind {
                        EntryKind::Directory => dirs.push(e.path),
                        EntryKind::File => {
                            if engine::decode::is_supported(&e.path) {
                                seqs.push(single_row(&e.path));
                            } else {
                                other.push(e.path);
                            }
                        }
                        _ => {}
                    }
                }
                for s in &l.sequences {
                    seqs.extend(rows_of(s));
                }
                seqs.sort_by(|a, b| natural_cmp(&a.pattern, &b.pattern));
                let lifted = seqs.len();
                let mut version_dirs = BTreeSet::new();
                for d in dirs.iter().filter(|d| is_version_dir(d)) {
                    let Ok(sub) = FileSequence::parse_directory(d, 2) else { continue };
                    let name = d.file_name().unwrap_or_default();
                    let before = seqs.len();
                    for e in sub.entries {
                        if !e.hidden
                            && matches!(e.kind, EntryKind::File)
                            && engine::decode::is_supported(&e.path)
                        {
                            seqs.push(single_row(&e.path));
                        }
                    }
                    for sq in &sub.sequences {
                        seqs.extend(rows_of(sq));
                    }
                    // The folder rides in the pattern, so the row says
                    // where it lives and its group key keeps the folder's
                    // version token.
                    for r in &mut seqs[before..] {
                        r.pattern = Path::new(name).join(&r.pattern).display().to_string();
                    }
                    if seqs.len() > before {
                        version_dirs.insert(d.clone());
                    }
                }
                Listing { dirs, seqs, other, lifted, version_dirs }
            });
        self.listing = Some(result);
        self.listing_gen += 1;
    }

    fn probe(&mut self, row_first: &PathBuf, frames: &[PathBuf]) -> &Probe {
        if !self.probes.contains_key(row_first) {
            // The FILE's format, not a guess: this row may be a png, a dpx
            // or a movie now, and "exr half" was true only while exr was
            // the only thing that loaded. "half" stays — every decoder in
            // the app lands its pixels in the same f16 RGBA buffer, which
            // is the part worth telling someone about here.
            let kind = ext_of(row_first).unwrap_or_else(|| "?".into());
            let mut in_file = 1u64;
            let format = match self.decoder.read_header(row_first) {
                Ok(i) => {
                    // The header is in hand: the width floor need not read it again.
                    self.dims.known.insert(row_first.clone(), Some((i.width, i.height)));
                    in_file = engine::decode::is_movie(row_first)
                        .then(|| self.decoder.frame_count(row_first).unwrap_or(1))
                        .unwrap_or(1);
                    if engine::decode::is_movie(row_first) {
                        self.dims.counts.insert(row_first.clone(), in_file);
                    }
                    let frames_in = (in_file > 1)
                        .then(|| format!(", {in_file} frames in the file"))
                        .unwrap_or_default();
                    format!("{}x{} {kind} half{frames_in}", i.width, i.height)
                }
                Err(e) => format!("unreadable: {e:#}"),
            };
            let total_bytes: u64 = frames
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len())
                .sum();
            self.probes
                .insert(row_first.clone(), Probe { format, total_bytes, frames: in_file });
        }
        &self.probes[row_first]
    }

    /// The frame the scrub position names, in a sequence of `len`.
    fn scrub_index(&self, len: usize) -> usize {
        scrub_index_of(self.scrub, len)
    }

    fn start_preview(&mut self, len: usize) {
        self.preview = Some(Preview {
            started: web_time::Instant::now(),
            from: self.scrub_index(len),
            len,
        });
    }

    fn stop_preview(&mut self) {
        self.preview = None;
    }

    /// Advance the transport and ask for the frame it now sits on.
    ///
    /// Returns whether a repaint is needed — the browser is an idle window
    /// most of the time, so without asking, playback would only advance
    /// when something else happened to cause a frame.
    fn pump_preview(&mut self) -> bool {
        let Some(p) = &self.preview else { return false };
        if p.len <= 1 {
            self.preview = None;
            return false;
        }
        self.scrub = p.frame() as f32 / (p.len - 1) as f32;
        true
    }

    fn request_thumb(&mut self, key: FrameKey) {
        if self.thumb_tex.as_ref().is_some_and(|(k, _)| *k == key)
            || self.thumb_inflight.as_ref() == Some(&key)
        {
            return;
        }
        self.thumb_wanted = Some(key.clone());
        if self.thumb_inflight.is_none() {
            self.thumb_inflight = Some(key.clone());
            spawn_thumb(self.decoder.clone(), key, THUMB_W, self.thumb_tx.clone());
        }
    }

    fn pump_thumbs(&mut self, ctx: &egui::Context) {
        while let Ok((key, img)) = self.thumb_rx.try_recv() {
            self.thumb_inflight = None;
            let tex = ctx.load_texture("seq-thumb", img, egui::TextureOptions::LINEAR);
            self.seen_frames.insert(key.clone());
            self.thumb_tex = Some((key, tex));
            // Latest-wins: if the user scrubbed on while we decoded, chase it.
            if let Some(want) = self.thumb_wanted.clone() {
                if self.thumb_tex.as_ref().is_none_or(|(k, _)| *k != want) {
                    self.thumb_inflight = Some(want.clone());
                    spawn_thumb(self.decoder.clone(), want, THUMB_W, self.thumb_tx.clone());
                }
            }
        }
    }


    fn inspector_ui(
        &mut self,
        ui: &mut egui::Ui,
        dir: &Path,
        row: &SeqRow,
        group: Option<&VersionGroup>,
        action: &mut BrowserAction,
    ) {
        let (pattern, frames) = (row.pattern.as_str(), &row.frames[..]);
        let (first, last, missing) = (row.first, row.last, row.missing);
        let probe_key = frames[0].clone();
        let (format, total_bytes, in_file) = {
            let p = self.probe(&probe_key, frames);
            (p.format.clone(), p.total_bytes, p.frames)
        };
        // Everything below counts in SCRUBBABLE frames, which for a movie
        // is what the header said and for a sequence is the file count.
        let len = row.scrub_len(in_file);
        let frame_at = |i: usize| row.frame_key(i);
        // One place asks for the inspector's picture, whatever moved the
        // position: a drag on the strip, the transport, or a new selection.
        let want = frame_at(self.scrub_index(len));
        self.request_thumb(want);
        ui.horizontal(|ui| {
            // Thumbnail with scrub strip under it.
            ui.vertical(|ui| {
                let thumb_size = egui::vec2(148.0, 84.0);
                let (rect, _) =
                    ui.allocate_exact_size(thumb_size, egui::Sense::hover());
                if let Some((_, tex)) = &self.thumb_tex {
                    egui::Image::from_texture(tex).paint_at(ui, fit_rect(rect, tex.aspect_ratio()));
                } else {
                    ui.painter().rect_filled(rect, 2.0, theme::raised_fill());
                }
                if self.thumb_inflight.is_some() {
                    ui.put(rect, egui::Spinner::new().size(14.0));
                }

                if scrub_strip(ui, egui::vec2(thumb_size.x, 12.0), row, &mut self.scrub) {
                    // A hand on the strip takes the transport back.
                    self.stop_preview();
                }

                // Transport. OPTIONAL by design: nothing decodes here until
                // it is pressed, so browsing a directory of 900-frame plates
                // stays as cheap as it was before there was a player.
                let cur = self.scrub_index(len);
                if transport_row(ui, self.preview.is_some(), len > 1 && row.readable, cur, len) {
                    if self.preview.is_some() {
                        self.stop_preview();
                    } else {
                        self.start_preview(len);
                    }
                }
            });

            // insp rows.
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.strong(pattern);
                    if let Some(g) = group {
                        if let Some(v) = version_menu(ui, "seq-insp-ver", g) {
                            self.version_pick.insert(g.key.clone(), v);
                            self.follow_group = Some(g.key.clone());
                        }
                    }
                });
                ui.weak(dir.display().to_string());
                ui.monospace(format!("format   {format}"));
                let range = if frames.len() == 1 {
                    "single frame".to_string()
                } else {
                    format!("{first}-{last}  ({} frames)", frames.len())
                };
                ui.monospace(format!("range    {range}"));
                if missing > 0 {
                    ui.colored_label(
                        theme::warn(),
                        format!("missing  {missing} frames"),
                    );
                }
                ui.monospace(format!(
                    "size     {:.1} MB  ({:.1} MB/frame)",
                    total_bytes as f64 / 1e6,
                    total_bytes as f64 / 1e6 / frames.len().max(1) as f64
                ));
                ui.weak(fmt_date_full(row.modified));
                if ui.button("use in Load").clicked() {
                    *action = BrowserAction::Pick {
                        display: dir.join(pattern).display().to_string(),
                        frames: frames.to_vec(),
                        missing,
                    };
                }
            });
        });
    }
}


/// The big viewer: Space on a clip. One clip, filling the screen, with a
/// transport of its own — Finder's Quick Look, because that is the hand
/// everyone already has. Frames come from `SeqBrowser::big`, the large
/// thumbnail cache, so putting it away and bringing it back is free.
struct QuickLook {
    dir: PathBuf,
    row: SeqRow,
    /// Scrubbable frames (a movie's probed count, a sequence's files).
    len: usize,
    scrub: f32,
    preview: Option<Preview>,
}

/// One line of the types sidebar.
struct ExtCount {
    /// Lower-case; "" for files with no extension.
    ext: String,
    /// Listing rows (a sequence is one row) and the files behind them.
    rows: usize,
    files: usize,
    supported: bool,
}

/// Tally `(extension, rows, files)` triples per extension: decodable types
/// first, then alphabetical, so the ticks that matter sit at the top.
fn count_exts(items: impl IntoIterator<Item = (Option<String>, usize, usize)>) -> Vec<ExtCount> {
    let mut map: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    for (ext, rows, files) in items {
        let e = map.entry(ext_key(ext.as_deref())).or_default();
        e.0 += rows;
        e.1 += files;
    }
    let mut out: Vec<ExtCount> = map
        .into_iter()
        .map(|(ext, (rows, files))| ExtCount {
            supported: !ext.is_empty() && is_supported_ext(&ext),
            ext,
            rows,
            files,
        })
        .collect();
    out.sort_by(|a, b| b.supported.cmp(&a.supported).then(a.ext.cmp(&b.ext)));
    out
}

/// The frame a 0..1 scrub position names, in a sequence of `len`.
fn scrub_index_of(scrub: f32, len: usize) -> usize {
    ((scrub * len.saturating_sub(1) as f32).round() as usize).min(len.saturating_sub(1))
}

/// Scrub strip: occupancy bar + frame picker in one. Present frames green
/// over a red base when there are gaps, so the holes show through; a hand
/// on it moves `scrub`. Returns whether it was touched, so the caller can
/// take the transport back from the player.
fn scrub_strip(ui: &mut egui::Ui, size: egui::Vec2, row: &SeqRow, scrub: &mut f32) -> bool {
    let (strip, resp) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());
    let painter = ui.painter();
    painter.rect_filled(strip, 2.0, theme::range_out());
    let span = (row.last - row.first).max(0) as f32 + 1.0;
    if row.missing > 0 {
        painter.rect_filled(
            strip,
            2.0,
            Color32::BLACK.blend(theme::error().gamma_multiply(0.55)),
        );
    }
    for &n in &row.numbers {
        let x0 = strip.min.x + (n - row.first) as f32 / span * strip.width();
        let w = (strip.width() / span).max(1.0);
        painter.rect_filled(
            egui::Rect::from_min_size(
                egui::pos2(x0, strip.min.y),
                egui::vec2(w, strip.height()),
            ),
            0.0,
            Color32::BLACK.blend(theme::ok().gamma_multiply(0.55)),
        );
    }
    let mut touched = false;
    if resp.dragged() || resp.clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            touched = true;
            *scrub = ((p.x - strip.min.x) / strip.width().max(1.0)).clamp(0.0, 1.0);
        }
    }
    let cx = strip.min.x + *scrub * strip.width();
    painter.vline(cx, strip.y_range(), egui::Stroke::new(2.0, theme::playhead()));
    touched
}

/// ▶/⏸ plus the frame counter. True when the button was pressed.
fn transport_row(ui: &mut egui::Ui, playing: bool, can_play: bool, cur: usize, len: usize) -> bool {
    let mut toggled = false;
    ui.horizontal(|ui| {
        let icon = if playing { "⏸" } else { "▶" };
        if ui
            .add_enabled(
                can_play,
                egui::Button::new(icon).min_size(egui::vec2(24.0, 18.0)),
            )
            .on_hover_text(if playing {
                "pause"
            } else {
                "play this footage here, without loading it into the comp"
            })
            .on_disabled_hover_text("a single frame has nothing to play")
            .clicked()
        {
            toggled = true;
        }
        ui.small(RichText::new(format!("{}/{len}", cur + 1)).monospace());
        if playing {
            ui.small(RichText::new(format!("{PREVIEW_FPS} fps")).weak());
        }
    });
    toggled
}

impl SeqBrowser {
    /// The type gate, base filter and sidebar ticks together.
    fn shows(&self, ext: Option<&str>) -> bool {
        shown(self.types, &self.excluded, ext)
    }

    /// Every extension in view with how much of it there is: the flat
    /// listing's rows and loose files, or the recursive search's hits plus
    /// the loose files it counted but did not list.
    fn ext_counts(&self) -> Vec<ExtCount> {
        if let Some(se) = &self.search {
            return count_exts(
                se.hits
                    .iter()
                    .map(|(_, r)| (r.ext.clone(), 1, r.frames.len()))
                    .chain(se.other.iter().map(|(e, n)| (Some(e.clone()), *n, *n))),
            );
        }
        match self.listing.as_ref().and_then(|l| l.as_ref().ok()) {
            Some(l) => count_exts(
                l.seqs
                    .iter()
                    .map(|r| (r.ext.clone(), 1, r.frames.len()))
                    .chain(l.other.iter().map(|p| (ext_of(p), 1, 1))),
            ),
            None => Vec::new(),
        }
    }

    /// The right sidebar: base filter and duration floor, then every
    /// extension found with a count and a tick. Untick `.png` and the pngs
    /// go — under either base filter, in the flat listing and the recursive
    /// search alike.
    fn types_sidebar(&mut self, ui: &mut egui::Ui) {
        ui.small(RichText::new("SHOW").weak());
        ui.horizontal_wrapped(|ui| {
            for t in [TypeFilter::Supported, TypeFilter::All] {
                let r = ui.selectable_label(self.types == t, t.label());
                let r = match t {
                    TypeFilter::Supported => r.on_hover_text(supported_hint()),
                    TypeFilter::All => r.on_hover_text(
                        "everything in the directory — files this build cannot decode list dimmed",
                    ),
                };
                if r.clicked() {
                    self.types = t;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.small("at least");
            ui.add(
                egui::DragValue::new(&mut self.min_frames).update_while_editing(false)
                    .speed(crate::widgets::rated(ui, 1.0))
                    .range(0..=100_000)
                    .suffix("f"),
            )
            .on_hover_text("minimum duration in frames — 0 shows everything");
        });
        ui.horizontal(|ui| {
            ui.small("at least");
            let r = ui.add(
                egui::DragValue::new(&mut self.min_width).update_while_editing(false)
                    .speed(crate::widgets::rated(ui, 8.0))
                    .range(0..=16_384)
                    .suffix("px wide"),
            )
            .on_hover_text(
                "minimum width in pixels — 0 shows everything. Read off each \
                 file's header on a worker; a clip stays listed until its size \
                 is known",
            );
            if r.changed() {
                self.resort = true;
            }
            let pending = self.min_width > 0 && !self.dims.inflight.is_empty();
            if pending {
                ui.spinner().on_hover_text(format!(
                    "measuring {} clips",
                    self.dims.inflight.len()
                ));
            }
        });
        ui.add_space(8.0);

        ui.small(RichText::new("TYPES").weak());
        let counts = self.ext_counts();
        let recursive = self.search.is_some();
        if counts.is_empty() {
            ui.weak("nothing found yet");
        }
        for c in &counts {
            // A type the base filter would hide anyway has nothing to untick.
            let listable = c.supported || self.types == TypeFilter::All;
            let mut on = listable && !self.excluded.contains(&c.ext);
            let label = if c.ext.is_empty() {
                "no extension".to_string()
            } else {
                format!(".{}", c.ext)
            };
            ui.horizontal(|ui| {
                let r = ui.add_enabled(listable, egui::Checkbox::new(&mut on, label));
                if r.changed() {
                    if on {
                        self.excluded.remove(&c.ext);
                    } else {
                        self.excluded.insert(c.ext.clone());
                    }
                }
                let why = if c.supported {
                    format!(
                        "{} decodable file{}",
                        c.files,
                        if c.files == 1 { "" } else { "s" }
                    )
                } else if !listable {
                    "not decodable — hidden under \"supported media\"".to_string()
                } else if recursive {
                    "not decodable — numbered sequences of it list dimmed; loose files are only counted"
                        .to_string()
                } else {
                    "not decodable — listed dimmed, never loadable".to_string()
                };
                r.on_hover_text(why);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let n = if c.files > c.rows {
                        format!("{} · {}f", c.rows, c.files)
                    } else {
                        c.rows.to_string()
                    };
                    ui.small(RichText::new(n).monospace().weak());
                });
            });
        }
        if counts.len() > 1 {
            ui.horizontal(|ui| {
                if ui.small_button("all").on_hover_text("tick every type").clicked() {
                    self.excluded.clear();
                }
                if ui.small_button("none").on_hover_text("untick every type").clicked() {
                    for c in &counts {
                        self.excluded.insert(c.ext.clone());
                    }
                }
            });
        }
        ui.add_space(8.0);

        ui.small(RichText::new("FOUND").weak());
        if let Some(se) = &self.search {
            let kept = se.hits.iter().filter(|(_, r)| self.keep(r)).count();
            ui.horizontal(|ui| {
                if se.running {
                    ui.spinner();
                }
                ui.small(
                    RichText::new(format!(
                        "{} directories walked{}",
                        se.scanned,
                        if se.running { " …" } else { "" }
                    ))
                    .weak(),
                );
            });
            ui.small(RichText::new(format!("{kept} of {} listed", se.hits.len())).weak());
        } else if let Some(l) = self.listing.as_ref().and_then(|l| l.as_ref().ok()) {
            let kept = l.seqs.iter().filter(|r| self.keep(r)).count();
            let single = l.seqs.iter().filter(|r| r.frames.len() == 1).count();
            ui.small(
                RichText::new(format!(
                    "{} folders · {} sequences · {} single files",
                    l.dirs.len(),
                    l.seqs.len() - single,
                    single
                ))
                .weak(),
            );
            if !l.other.is_empty() {
                ui.small(RichText::new(format!("{} other files", l.other.len())).weak());
            }
            ui.small(RichText::new(format!("{kept} of {} listed", l.seqs.len())).weak());
        }
    }


    fn open_quicklook(&mut self, dir: PathBuf, row: SeqRow) {
        if !row.readable {
            return;
        }
        // A movie's length is a header read; it is paid here, on demand,
        // never for a whole directory of clips.
        let probed = self.probe(&row.frames[0].clone(), &row.frames).frames;
        let len = row.scrub_len(probed);
        self.quicklook = Some(QuickLook { dir, row, len, scrub: 0.0, preview: None });
    }

    /// The big viewer, when it is open. Space (or Esc, or a click on the
    /// backdrop, or ×) puts it away; ←/→ step frames; ▶ plays it here.
    fn quicklook_ui(&mut self, ctx: &egui::Context, action: &mut BrowserAction) {
        let Some(mut ql) = self.quicklook.take() else { return };
        // Position is derived from wall time, like the inspector's.
        if let Some(p) = &ql.preview {
            if ql.len <= 1 {
                ql.preview = None;
            } else {
                ql.scrub = p.frame() as f32 / (ql.len - 1) as f32;
                ctx.request_repaint();
            }
        }
        if !ctx.egui_wants_keyboard_input() && ql.len > 1 {
            let (l, r) = ctx.input(|i| {
                (
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                )
            });
            if l || r {
                let i = scrub_index_of(ql.scrub, ql.len) as i64 + if r { 1 } else { -1 };
                let i = i.rem_euclid(ql.len as i64) as usize;
                ql.scrub = i as f32 / (ql.len - 1) as f32;
                // A hand on the frame is a hand on the strip.
                ql.preview = None;
            }
        }
        let i = scrub_index_of(ql.scrub, ql.len);
        let key = ql.row.frame_key(i);
        // Read-ahead while playing, so the workers run a few frames in
        // front of the transport instead of chasing it.
        if ql.preview.is_some() {
            for k in 1..=3 {
                let _ = self.big.get(&ql.row.frame_key((i + k) % ql.len));
            }
        }
        // The big frame; until it lands, the tile-sized one (right frame,
        // soft), then the nearest big one (sharp, a frame off).
        let tex = self
            .big
            .get(&key)
            .or_else(|| self.thumbs.want(&Src::of(&ql.row), i as u32, 256, 0.0).map(|s| s.tex))
            .or_else(|| self.big.nearest(&key));
        let busy = self.big.pending(&key);
        let display = ql.dir.join(&ql.row.pattern).display().to_string();
        let binned = self.bin.iter().any(|b| b.display == display);
        let screen = ctx.content_rect();
        let size = (screen.size() - egui::vec2(48.0, 48.0)).max(egui::vec2(240.0, 180.0));
        let (mut close, mut toggle, mut bin, mut use_it) = (false, false, false, false);
        let modal = egui::Modal::new(egui::Id::new("seq-quicklook"))
            .backdrop_color(Color32::from_black_alpha(220))
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| {
                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                let bar_h = 58.0;
                let img = egui::Rect::from_min_max(
                    rect.min,
                    egui::pos2(rect.max.x, rect.max.y - bar_h),
                );
                // Opaque under the whole card: the picture AND the bar, so
                // the browser dimmed behind does not bleed into the strip.
                ui.painter().rect_filled(rect, 4.0, Color32::BLACK);
                if let Some(t) = &tex {
                    egui::Image::from_texture(t).paint_at(ui, fit_rect(img, t.aspect_ratio()));
                }
                if busy {
                    ui.put(
                        egui::Rect::from_center_size(img.center(), egui::vec2(28.0, 28.0)),
                        egui::Spinner::new().size(24.0),
                    );
                }
                let p = ui.painter().with_clip_rect(img);
                p.text(
                    img.left_top() + egui::vec2(12.0, 10.0),
                    egui::Align2::LEFT_TOP,
                    &ql.row.pattern,
                    egui::FontId::proportional(14.0),
                    theme::text_strong(),
                );
                p.text(
                    img.left_top() + egui::vec2(12.0, 30.0),
                    egui::Align2::LEFT_TOP,
                    ql.dir.display().to_string(),
                    egui::FontId::monospace(10.0),
                    theme::text(),
                );
                p.text(
                    img.right_top() + egui::vec2(-12.0, 10.0),
                    egui::Align2::RIGHT_TOP,
                    format!("{}/{}", i + 1, ql.len),
                    egui::FontId::monospace(12.0),
                    theme::text_strong(),
                );
                let bar = egui::Rect::from_min_max(egui::pos2(rect.min.x, img.max.y), rect.max)
                    .shrink2(egui::vec2(8.0, 6.0));
                ui.scope_builder(egui::UiBuilder::new().max_rect(bar), |ui| {
                    ui.vertical(|ui| {
                        if scrub_strip(ui, egui::vec2(bar.width(), 14.0), &ql.row, &mut ql.scrub)
                        {
                            ql.preview = None;
                        }
                        ui.horizontal(|ui| {
                            toggle = transport_row(ui, ql.preview.is_some(), ql.len > 1, i, ql.len);
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.button("×").on_hover_text("close  (space / esc)").clicked()
                                    {
                                        close = true;
                                    }
                                    if ui
                                        .button("use")
                                        .on_hover_text("load this clip, and close the browser")
                                        .clicked()
                                    {
                                        use_it = true;
                                    }
                                    if ui
                                        .add_enabled(!binned, egui::Button::new("+ bin"))
                                        .on_disabled_hover_text("already in the bin")
                                        .clicked()
                                    {
                                        bin = true;
                                    }
                                    // ASCII on purpose: the UI font has no arrow glyphs.
                                    ui.small(RichText::new("arrow keys step · space closes").weak());
                                },
                            );
                        });
                    });
                });
            });
        if toggle {
            ql.preview = match ql.preview {
                Some(_) => None,
                None => Some(Preview { started: web_time::Instant::now(), from: i, len: ql.len }),
            };
        }
        if bin {
            self.add_to_bin(&ql.dir, &ql.row);
        }
        if use_it {
            *action = pick(&ql.dir, &ql.row);
            close = true;
        }
        if close || modal.should_close() {
            return;
        }
        self.quicklook = Some(ql);
    }
}

fn pick_of(dir: &Path, s: &SeqRow) -> Picked {
    Picked {
        display: dir.join(&s.pattern).display().to_string(),
        frames: s.frames.to_vec(),
        missing: s.missing,
    }
}

fn pick(dir: &Path, s: &SeqRow) -> BrowserAction {
    let p = pick_of(dir, s);
    BrowserAction::Pick {
        display: p.display,
        frames: p.frames,
        missing: p.missing,
    }
}

/// A loose exr is a one-frame sequence.
fn ext_of(p: &Path) -> Option<String> {
    p.extension().map(|e| e.to_string_lossy().to_lowercase())
}

fn single_row(path: &Path) -> SeqRow {
    let ext = ext_of(path);
    let modified = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
    SeqRow {
        pattern: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        frames: Arc::from(vec![path.to_path_buf()]),
        numbers: vec![0],
        first: 0,
        last: 0,
        missing: 0,
        readable: ext
            .as_deref()
            .is_some_and(|e| SUPPORTED.iter().any(|s| s.eq_ignore_ascii_case(e))),
        ext,
        modified,
        movie: engine::decode::is_movie(path),
    }
}

/// A scanned group as browser rows. Normally that is one row — a numbered
/// image sequence — but MOVIES that happen to be numbered (`shot_01.mov`,
/// `shot_02.mov`) are not frames of anything: each file is its own clip
/// with its own length, so they come back as one row EACH.
fn rows_of(s: &FileSequence) -> Vec<SeqRow> {
    let movies = s
        .items()
        .iter()
        .any(|i| engine::decode::is_movie(&i.path()));
    if movies {
        return s.items().iter().map(|i| single_row(&i.path())).collect();
    }
    vec![row_of(s)]
}

fn row_of(s: &FileSequence) -> SeqRow {
    let ext = s.extension().ok().map(|e| e.to_lowercase());
    let frames: Vec<PathBuf> = s.items().iter().map(|i| i.path()).collect();
    let modified = seq_modified(&frames);
    SeqRow {
        pattern: s.sequence_string().unwrap_or_else(|_| "<mixed sequence>".into()),
        frames: frames.into(),
        numbers: s.items().iter().map(|i| i.frame_number()).collect(),
        first: s.first_frame(),
        last: s.last_frame(),
        missing: (s.frame_count() as i64 - s.actual_frame_count() as i64).max(0) as u64,
        readable: ext
            .as_deref()
            .is_some_and(|e| SUPPORTED.iter().any(|s| s.eq_ignore_ascii_case(e))),
        ext,
        modified,
        // `rows_of` splits movies into their own single rows, so a grouped
        // numbered sequence is never one.
        movie: false,
    }
}

/// Decode one frame on the rayon pool, downsample + tonemap to a thumbnail.
/// Shared with the graph's TV thumbnails (`crate::tv`): one decode + tone
/// path for every thumbnail in the app.
pub(crate) fn spawn_thumb(
    decoder: Arc<dyn ExrDecoder>,
    key: FrameKey,
    max_w: usize,
    tx: Sender<(FrameKey, egui::ColorImage)>,
) {
    rayon::spawn(move || {
        let img = decode_thumb(&*decoder, &key, |w, h| {
            let tw = w.min(max_w).max(1);
            (tw, (h * tw / w).max(1))
        });
        if let Some(img) = img {
            let _ = tx.send((key, img));
        }
    });
}

/// Decode one frame and shrink + tonemap it to the size `fit` picks from
/// the frame's own (width, height).
fn decode_thumb(decoder: &dyn ExrDecoder, key: &FrameKey, fit: impl Fn(usize, usize) -> (usize, usize)) -> Option<egui::ColorImage> {
    let (path, frame) = (&key.0, key.1);
    let info = decoder.read_header(path).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    if w == 0 || h == 0 {
        return None;
    }
    let stride = w * 4;
    let mut buf = vec![f16::ZERO; stride * h];
    decoder.decode_frame_into(path, info, stride, None, frame, &mut buf).ok()?;
    let (tw, th) = fit(w, h);
    let mut rgba = vec![0u8; tw * th * 4];
    let tone = |x: f32| -> u8 {
        // Same ACES-ish fit + sRGB encode as the viewer tail.
        let x = x.max(0.0);
        let t = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
        let t = t.clamp(0.0, 1.0);
        let e = if t <= 0.0031308 { t * 12.92 } else { 1.055 * t.powf(1.0 / 2.4) - 0.055 };
        (e * 255.0 + 0.5) as u8
    };
    for y in 0..th {
        let sy = y * h / th;
        for x in 0..tw {
            let sx = x * w / tw;
            let s = sy * stride + sx * 4;
            let d = (y * tw + x) * 4;
            rgba[d] = tone(buf[s].to_f32());
            rgba[d + 1] = tone(buf[s + 1].to_f32());
            rgba[d + 2] = tone(buf[s + 2].to_f32());
            rgba[d + 3] = 255;
        }
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([tw, th], &rgba))
}

#[cfg(test)]
mod types_tests {
    use super::*;

    /// Unticking a type in the sidebar hides it whatever the base filter
    /// says, and only that type.
    #[test]
    fn unticking_an_extension_hides_it_under_either_base_filter() {
        let mut ex = BTreeSet::new();
        assert!(shown(TypeFilter::Supported, &ex, Some("png")));
        ex.insert("png".into());
        assert!(!shown(TypeFilter::Supported, &ex, Some("png")));
        assert!(!shown(TypeFilter::All, &ex, Some("PNG")), "case-insensitive");
        assert!(shown(TypeFilter::Supported, &ex, Some("exr")));
        assert!(!shown(TypeFilter::Supported, &ex, Some("txt")), "base filter still applies");
        assert!(shown(TypeFilter::All, &ex, None));
        ex.insert(String::new());
        assert!(!shown(TypeFilter::All, &ex, None), "extensionless files key as \"\"");
    }

    /// The width floor hides what is too small and nothing it cannot
    /// measure — an unread or unreadable header passes.
    #[test]
    fn the_width_floor_only_hides_what_it_has_measured() {
        assert!(width_ok(1920, Some(&Some((2048, 1080)))));
        assert!(width_ok(2048, Some(&Some((2048, 1080)))));
        assert!(!width_ok(4096, Some(&Some((2048, 1080)))));
        assert!(width_ok(4096, None), "not read yet stays listed");
        assert!(width_ok(4096, Some(&None)), "unreadable stays listed");
    }

    #[test]
    fn the_sidebar_counts_rows_and_files_per_extension() {
        let counts = count_exts([
            (Some("exr".to_string()), 1, 100),
            (Some("EXR".to_string()), 1, 24),
            (Some("png".to_string()), 1, 1),
            (None, 3, 3),
        ]);
        let names: Vec<&str> = counts.iter().map(|c| c.ext.as_str()).collect();
        assert_eq!(names, ["exr", "png", ""], "decodable first, then a-z");
        assert_eq!((counts[0].rows, counts[0].files, counts[0].supported), (2, 124, true));
        assert_eq!((counts[2].rows, counts[2].files, counts[2].supported), (3, 3, false));
    }
}


#[cfg(test)]
mod sort_tests {
    use super::*;
    use web_time::Duration;

    fn row(pattern: &str, first: i32, last: i32, secs: u64) -> SeqRow {
        SeqRow {
            pattern: pattern.into(),
            frames: vec![PathBuf::from(pattern)].into(),
            numbers: vec![first],
            first,
            last,
            missing: 0,
            readable: true,
            ext: Some("exr".into()),
            modified: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(secs)),
            movie: false,
        }
    }

    fn order(rows: &mut Vec<SeqRow>, key: SortKey, desc: bool) -> Vec<String> {
        rows.sort_by(|a, b| row_cmp(a, b, key, desc));
        rows.iter().map(|r| r.pattern.clone()).collect()
    }

    #[test]
    fn names_sort_naturally() {
        let mut rows = vec![
            row("shot10.####.exr", 1, 1, 0),
            row("Shot2.####.exr", 1, 1, 0),
            row("shot1.####.exr", 1, 1, 0),
            row("shot02b.####.exr", 1, 1, 0),
        ];
        assert_eq!(
            order(&mut rows, SortKey::Name, false),
            ["shot1.####.exr", "Shot2.####.exr", "shot02b.####.exr", "shot10.####.exr"]
        );
        use std::cmp::Ordering::*;
        assert_eq!(natural_cmp("a99", "a100"), Less);
        assert_eq!(natural_cmp("a007", "a7"), Less); // tie broken on bytes, never Equal
        assert_eq!(natural_cmp("a", "a"), Equal);
        // A run longer than any integer still compares.
        assert_eq!(natural_cmp("x123456789012345678901234", "x99"), Greater);
    }

    #[test]
    fn a_version_is_a_v_number_that_starts_a_word() {
        assert_eq!(
            version_of("shot_010_comp_v003.####.exr"),
            Some((3, "shot_010_comp_v#.####.exr".into()))
        );
        // Every token goes into the key, the LAST one is the version, and
        // a folder counts — either separator.
        assert_eq!(version_of("comp/v2/shot_V12.exr"), Some((12, "comp/v#/shot_v#.exr".into())));
        let win = r"V:\projects\laMer\lame05_p02\sh030\publish\render\renderLgtRedshift_ropUTL_BG";
        let a = format!(r"{win}\v051\lame05_p02_sh030_renderLgtRedshift_ropUTL_BG_v051.####.exr");
        let b = format!(r"{win}\v049\lame05_p02_sh030_renderLgtRedshift_ropUTL_BG_v049.####.exr");
        let (va, ka) = version_of(&a).unwrap();
        let (vb, kb) = version_of(&b).unwrap();
        assert_eq!((va, vb), (51, 49));
        assert_eq!(ka, kb);
        assert_eq!(version_of("comp/v2/shot.####.exr"), Some((2, "comp/v#/shot.####.exr".into())));
        // Inside a word, or no digits: not a version.
        assert_eq!(version_of("dev12_plate.####.exr"), None);
        assert_eq!(version_of("plate_vfx.exr"), None);
        assert_eq!(version_of("v_plate.exr"), None);
    }

    #[test]
    fn versions_collapse_to_the_latest_or_the_pick() {
        let items = || {
            [
                "a_v001.####.exr",
                "a_v003.####.exr",
                "a_v002.####.exr",
                "b_v001.####.exr",
                "plate.####.exr",
                "a_v001.####.dpx",
            ]
            .iter()
            .enumerate()
            .map(|(i, p)| (i, p.to_string()))
            .collect::<Vec<_>>()
        };
        let (hidden, groups) = collapse_versions(items(), &HashMap::new());
        // a's v001 and v002 hide behind v003; b (one version), the plate and
        // the .dpx (another key) stay as they are.
        assert_eq!(hidden, [0, 2].into_iter().collect());
        assert_eq!(
            groups.get(&1),
            Some(&VersionGroup { key: "a_v#.####.exr".into(), versions: vec![1, 2, 3], shown: 3 })
        );
        assert_eq!(groups.len(), 1);
        // A pick shows that version instead.
        let picks = HashMap::from([("a_v#.####.exr".to_string(), 1)]);
        let (hidden, groups) = collapse_versions(items(), &picks);
        assert_eq!(hidden, [1, 2].into_iter().collect());
        assert_eq!(groups[&0].shown, 1);
        // A pick that is no longer listed falls back to the latest.
        let picks = HashMap::from([("a_v#.####.exr".to_string(), 9)]);
        let (hidden, _) = collapse_versions(items(), &picks);
        assert_eq!(hidden, [0, 2].into_iter().collect());
    }

    #[test]
    fn duration_is_the_span_not_the_file_count() {
        // A gappy 1-100 sequence still occupies 100 frames.
        assert_eq!(row("a", 1, 100, 0).duration(), 100);
        assert_eq!(row("a", 0, 0, 0).duration(), 1);
    }

    #[test]
    fn sorts_by_each_column_both_ways() {
        let mut rows = vec![
            row("b", 1, 10, 300),  // 10f, newest
            row("a", 1, 50, 100),  // 50f, oldest
            row("c", 1, 20, 200),  // 20f
        ];
        assert_eq!(order(&mut rows, SortKey::Name, false), ["a", "b", "c"]);
        assert_eq!(order(&mut rows, SortKey::Name, true), ["c", "b", "a"]);
        assert_eq!(order(&mut rows, SortKey::Date, true), ["b", "c", "a"]);
        assert_eq!(order(&mut rows, SortKey::Date, false), ["a", "c", "b"]);
        assert_eq!(order(&mut rows, SortKey::Duration, true), ["a", "c", "b"]);
        assert_eq!(order(&mut rows, SortKey::Duration, false), ["b", "c", "a"]);
    }

    #[test]
    fn equal_keys_fall_back_to_name() {
        // Otherwise rows with the same date shuffle between frames.
        let mut rows = vec![row("z", 1, 5, 7), row("a", 1, 5, 7)];
        assert_eq!(order(&mut rows, SortKey::Date, true), ["z", "a"]);
        assert_eq!(order(&mut rows, SortKey::Duration, false), ["a", "z"]);
    }

    #[test]
    fn undated_rows_sort_oldest_and_never_panic() {
        let mut undated = row("u", 1, 5, 0);
        undated.modified = None;
        let mut rows = vec![row("d", 1, 5, 100), undated];
        assert_eq!(order(&mut rows, SortKey::Date, false), ["u", "d"]);
    }

    #[test]
    fn date_formats_from_unix_seconds() {
        let at = |s: u64| Some(SystemTime::UNIX_EPOCH + Duration::from_secs(s));
        assert_eq!(fmt_date(at(0)), "1970-01-01");
        assert_eq!(fmt_date(at(1_700_000_000)), "2023-11-14");
        assert_eq!(
            fmt_date_full(at(1_700_000_000)),
            "modified 2023-11-14 22:13 UTC"
        );
        // Leap day, and a pre-epoch stamp (clock skew, exotic archives).
        assert_eq!(fmt_date(at(1_709_164_800)), "2024-02-29");
        assert_eq!(fmt_date(Some(SystemTime::UNIX_EPOCH - Duration::from_secs(1))), "1969-12-31");
        assert_eq!(fmt_date(None), "—");
    }
}


#[cfg(test)]
mod filter_tests {
    use super::{supported_hint, TypeFilter, SUPPORTED};

    /// The default filter tracks the DECODER's list, not a copy of it. The
    /// failure this catches is the one that already happened once: formats
    /// were added to `SUPPORTED_EXTS` while the browser went on advertising
    /// and showing only EXR.
    #[test]
    fn the_default_filter_follows_the_decoder() {
        let f = TypeFilter::Supported;
        for e in ["exr", "EXR", "dpx", "png", "jpg", "jpeg"] {
            assert!(f.accepts(Some(e)), "{e} decodes but the filter hides it");
        }
        for e in ["psd", "txt", "tif", "r3d"] {
            assert!(!f.accepts(None) && !f.accepts(Some(e)), "{e} does not decode");
        }
        // Movies only when this build can actually read them — offering a
        // format we cannot decode is worse than not listing it.
        for e in engine::decode::MOVIE_EXTS {
            assert!(f.accepts(Some(e)), "{e} is in MOVIE_EXTS but filtered out");
        }
        assert_eq!(
            engine::decode::MOVIE_EXTS.is_empty(),
            !f.accepts(Some("mov")),
            "the mov filter must agree with whether this build has movies"
        );
    }

    /// The label is a fixed phrase (eleven extensions do not fit a 120pt
    /// combo) but the HINT has to be exact, or "supported media" is just a
    /// word. It names every extension the decoder takes, and no others.
    #[test]
    fn the_hint_names_what_actually_loads() {
        assert_eq!(TypeFilter::Supported.label(), "supported media");
        let hint = supported_hint();
        for e in SUPPORTED {
            assert!(hint.contains(&format!(".{e}")), "hint omits .{e}: {hint}");
        }
        assert!(!hint.contains(".psd"), "hint claims a format we cannot read");
        // Whichever way this build was compiled, the hint explains what the
        // filter will do with a movie rather than leaving it a mystery.
        assert!(hint.contains("Movies"), "{hint}");
        assert_eq!(
            engine::decode::MOVIE_EXTS.is_empty(),
            hint.contains("`movie` feature"),
            "a build without movies must say so; one with them must not"
        );
    }
}

#[cfg(test)]
mod action_bar_tests {
    use super::action_row;

    /// The action bar must be the SAME height on frame 20 as on frame 2.
    ///
    /// It was not: `with_layout(right_to_left(Align::Center))` centres by
    /// claiming all the height available to it, and inside a content-sized
    /// bottom panel that available height comes from the panel's own size
    /// last frame. Each frame the panel grew by a row (~7pt) and fed itself
    /// a bigger number, so a browser left open long enough ended up as one
    /// giant button with the listing squeezed into a sliver at the top.
    /// Growth per frame is what this measures — a single frame looks fine.
    #[test]
    fn the_commit_row_does_not_grow_each_frame() {
        let ctx = egui::Context::default();
        let mut heights = Vec::new();
        for _ in 0..20 {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::pos2(0.0, 0.0),
                    egui::vec2(1400.0, 900.0),
                )),
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ctx| {
                egui::Window::new("sequences")
                    .default_size([760.0, 540.0])
                    .show(ctx, |ui| {
                        let bar = egui::Panel::bottom(egui::Id::new("seq-actions"))
                            .resizable(false)
                            .show(ui, |ui| {
                                action_row(ui, 1, 1);
                            });
                        heights.push(bar.response.rect.height());
                        // A scroll area below is what the panel steals from,
                        // and it is what made the runaway visible.
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            for i in 0..8 {
                                ui.label(format!("row {i}"));
                            }
                        });
                    });
            });
        }
        let settled = &heights[2..];
        let first = settled[0];
        for (i, h) in settled.iter().enumerate() {
            assert!(
                (h - first).abs() < 0.5,
                "action bar grew: frame {} is {h}pt, frame 2 was {first}pt (all: {heights:?})",
                i + 2
            );
        }
        // And it is a ROW, not a region: a button plus its padding, nowhere
        // near the 540pt window it used to fill.
        assert!(first > 10.0 && first < 80.0, "implausible row height {first}pt");
    }
}

#[cfg(test)]
mod transport_tests {
    use super::*;

    /// Headless: select the plate, press play, and the transport's position
    /// advances on the wall clock while the browser keeps drawing frames.
    /// (`--test-scrub` covers the same with real decodes; this one runs in
    /// `cargo test` and points at the browser rather than the player.)
    #[test]
    fn the_preview_transport_advances_while_the_window_draws() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/2k_zip_bg");
        if !dir.is_dir() {
            eprintln!("SKIP: no data/2k_zip_bg");
            return;
        }
        let decoder: Arc<dyn ExrDecoder> = Arc::new(engine::decode::AnyDecoder::new(true));
        let mut b = SeqBrowser::new(None, dir, decoder);
        let ctx = egui::Context::default();
        let mut pinned = Vec::new();
        let mut frame = |b: &mut SeqBrowser| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1400.0, 900.0),
                )),
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ctx| {
                let _ = b.show(ctx, &mut pinned, &[]);
            });
        };
        frame(&mut b);
        let (len, playing) = b.test_select_and_len(0);
        assert_eq!(len, 24);
        assert!(!playing);
        b.test_play(len);
        let t0 = web_time::Instant::now();
        let mut at = 0;
        while t0.elapsed().as_millis() < 600 {
            std::thread::sleep(web_time::Duration::from_millis(10));
            frame(&mut b);
            let (i, playing) = b.test_preview_state(len);
            assert!(playing, "transport stopped on its own at {:?}", t0.elapsed());
            at = at.max(i);
        }
        assert!(at > 0, "after {:?} the transport never left frame 0", t0.elapsed());
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;

    /// `cargo test -p app browser_render -- --ignored --nocapture` renders
    /// the L-browser over `data/2k_zip_bg` — full-window layout, the types
    /// sidebar, a selected row's inspector — to `target/browser-gpu.png`,
    /// then presses Space and renders the quick look to
    /// `target/browser-quicklook-gpu.png`. GPU + test media, so ignored.
    #[test]
    #[ignore]
    fn browser_render() {
        let gpu = match engine::GpuContext::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("SKIP browser_render: no adapter ({e})");
                return;
            }
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../data");
        if !root.join("2k_zip_bg").is_dir() {
            eprintln!("SKIP browser_render: no data/2k_zip_bg");
            return;
        }
        let decoder: Arc<dyn ExrDecoder> = Arc::new(engine::decode::AnyDecoder::new(true));
        let dir = root.join("2k_zip_bg").canonicalize().unwrap();
        let mut b = SeqBrowser::new(None, dir.clone(), decoder);
        let mut pinned = vec![root.canonicalize().unwrap()];
        let recent = vec![dir.clone()];
        let (w_pt, h_pt, ppp) = (1400.0f32, 900.0f32, 2.0f32);
        let ctx = egui::Context::default();
        ctx.set_theme(egui::Theme::Dark);
        theme::sync(&ctx);
        let mut deltas = Vec::new();
        let mut frame = |events: Vec<egui::Event>, b: &mut SeqBrowser, deltas: &mut Vec<(egui::TextureId, egui::epaint::ImageDelta)>| {
            let mut input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(w_pt, h_pt),
                )),
                events,
                ..Default::default()
            };
            input
                .viewports
                .values_mut()
                .for_each(|v| v.native_pixels_per_point = Some(ppp));
            let out = ctx.run_ui(input, |ctx| {
                let _ = b.show(ctx, &mut pinned, &recent);
            });
            deltas.extend(out.textures_delta.set.iter().flat_map(|(id, ds)| ds.iter().map(move |d| (*id, d.clone()))));
            out
        };
        // Settle, select the first row, wait for its thumbnails.
        let mut out = frame(vec![], &mut b, &mut deltas);
        b.test_select(0);
        let t0 = web_time::Instant::now();
        while t0.elapsed().as_secs() < 8 && !(b.test_status().1 && b.test_counts().2 >= 1) {
            std::thread::sleep(web_time::Duration::from_millis(30));
            out = frame(vec![], &mut b, &mut deltas);
        }
        assert!(b.test_status().1, "inspector thumbnail never decoded");
        // A Window's area id is `Id::new(title.text())`, an Option<str>.
        let win = ctx
            .memory(|m| m.area_rect(egui::Id::new(Some("sequences — collect into bin"))))
            .expect("browser window rect");
        assert!(
            win.width() >= w_pt * 0.9 && win.height() >= h_pt * 0.9,
            "window is not full screen by default: {win:?}"
        );
        let counts = b.ext_counts();
        assert_eq!(counts.len(), 1, "one type in 2k_zip_bg: {:?}", counts.iter().map(|c| &c.ext).collect::<Vec<_>>());
        assert_eq!((counts[0].ext.as_str(), counts[0].rows, counts[0].files), ("exr", 1, 24));
        write_png(&gpu, &ctx, out, &deltas, "browser-gpu", w_pt, h_pt, ppp);

        // Space on the selected row: the quick look comes up, and its big
        // frame lands.
        let space = |pressed: bool| egui::Event::Key {
            key: egui::Key::Space,
            physical_key: None,
            pressed,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        };
        let _ = frame(vec![space(true)], &mut b, &mut deltas);
        assert!(b.quicklook.is_some(), "Space on a selected row opens the quick look");
        out = frame(vec![space(false)], &mut b, &mut deltas);
        let t0 = web_time::Instant::now();
        while t0.elapsed().as_secs() < 8 && b.big.tex.is_empty() {
            std::thread::sleep(web_time::Duration::from_millis(30));
            out = frame(vec![], &mut b, &mut deltas);
        }
        assert!(!b.big.tex.is_empty(), "big frame never decoded");
        assert!(b.quicklook.is_some(), "quick look closed by itself");
        write_png(&gpu, &ctx, out, &deltas, "browser-quicklook-gpu", w_pt, h_pt, ppp);
        // Space again puts it away; the browser is still there.
        let _ = frame(vec![space(true)], &mut b, &mut deltas);
        assert!(b.quicklook.is_none(), "Space again closes the quick look");
        // The width floor: 2048-wide plates survive 2048, not 4096, and a
        // row stays listed until its header has been read.
        b.test_set_min_width(4096);
        let t0 = web_time::Instant::now();
        let _ = frame(vec![], &mut b, &mut deltas);
        while t0.elapsed().as_secs() < 8 && b.test_dims_pending() > 0 {
            std::thread::sleep(web_time::Duration::from_millis(20));
            let _ = frame(vec![], &mut b, &mut deltas);
        }
        assert_eq!(b.test_counts().0, 0, "a 2048-wide plate listed under a 4096 floor");
        b.test_set_min_width(2048);
        assert_eq!(b.test_counts().0, 1, "a 2048-wide plate hidden under a 2048 floor");
        b.test_set_min_width(0);
        // A floor that hid the selected row dropped it from the selection
        // (as seq_manip does); select it again.
        let _ = frame(vec![], &mut b, &mut deltas);
        b.test_select(0);
        // The selection is the import list's feeder: one selected row,
        // "add selected to bin" puts exactly it in the bin.
        assert_eq!(b.selection.len(), 1, "test_select selects one row");
        b.bin_selection();
        assert_eq!(b.bin.len(), 1, "add-selected binned the selected row");
        b.bin_selection();
        assert_eq!(b.bin.len(), 1, "adding again does not duplicate");
        // Unticking the only type empties the listing.
        b.excluded.insert("exr".into());
        assert_eq!(b.test_counts().0, 0, "unticked .exr still listed");
        b.excluded.clear();
        assert_eq!(b.test_counts().0, 1);
    }

    fn write_png(
        gpu: &engine::GpuContext,
        ctx: &egui::Context,
        out: egui::FullOutput,
        deltas: &[(egui::TextureId, egui::epaint::ImageDelta)],
        name: &str,
        w_pt: f32,
        h_pt: f32,
        ppp: f32,
    ) {
        let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
        let (w, h) = ((w_pt * ppp) as u32, (h_pt * ppp) as u32);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("browser-render"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let mut renderer =
            egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());
        for (id, d) in deltas {
            renderer.update_texture(&gpu.device, &gpu.queue, *id, d);
        }
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: ppp };
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let user = renderer.update_buffers(&gpu.device, &gpu.queue, &mut enc, &prims, &screen);
        {
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("browser-render"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.08,
                                g: 0.08,
                                b: 0.09,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            renderer.render(&mut rp, &prims, &screen);
        }
        let unpadded = w as usize * 4;
        let padded = unpadded.div_ceil(256) * 256;
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("browser-render-rb"),
            size: (padded * h as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        gpu.queue.submit(user.into_iter().chain([enc.finish()]));
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
        {
            let data = buf.slice(..).get_mapped_range().expect("a mapped buffer");
            for y in 0..h as usize {
                for px in data[y * padded..y * padded + unpadded].chunks_exact(4) {
                    rgb.extend_from_slice(&px[..3]);
                }
            }
        }
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../target/{name}.png"));
        let file = std::fs::File::create(&path).unwrap();
        let mut png = png::Encoder::new(std::io::BufWriter::new(file), w, h);
        png.set_color(png::ColorType::Rgb);
        png.set_depth(png::BitDepth::Eight);
        png.write_header().unwrap().write_image_data(&rgb).unwrap();
        eprintln!("wrote {}", path.display());
    }
}

#[cfg(test)]
mod search_sort_tests {
    use super::*;
    use std::time::{Duration, UNIX_EPOCH};

    /// A recursive search sorted by date is ONE list across every
    /// directory, newest first — not a list per folder, not by name.
    #[test]
    fn a_recursive_search_sorts_by_date_across_directories() {
        let root = std::env::temp_dir().join(format!("wob-search-sort-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let put = |rel: &str, secs: u64| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            let f = std::fs::File::create(&p).unwrap();
            f.set_modified(UNIX_EPOCH + Duration::from_secs(secs)).unwrap();
        };
        put("a/old.0001.exr", 1_000);
        put("a/old.0002.exr", 1_000);
        put("a/z_newest.exr", 9_000);
        put("b/new.0001.exr", 5_000);
        put("b/new.0002.exr", 5_000);
        put("b/c/mid.exr", 3_000);
        put("top.exr", 4_000);
        let decoder: Arc<dyn ExrDecoder> = Arc::new(engine::decode::AnyDecoder::new(true));
        let mut b = SeqBrowser::new(None, root.clone(), decoder);
        let ctx = egui::Context::default();
        let mut pinned = Vec::new();
        let mut frame = |b: &mut SeqBrowser| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1400.0, 900.0),
                )),
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ctx| {
                let _ = b.show(ctx, &mut pinned, &[]);
            });
        };
        frame(&mut b);
        b.test_set_sort(SortKey::Date, true);
        b.test_search(4);
        let t0 = web_time::Instant::now();
        while t0.elapsed().as_secs() < 5 && b.test_search_status().2 {
            std::thread::sleep(Duration::from_millis(10));
            frame(&mut b);
        }
        frame(&mut b);
        let want = ["z_newest.exr", "new.####.exr", "top.exr", "mid.exr", "old.####.exr"];
        let names = |b: &SeqBrowser| -> Vec<String> {
            b.search.as_ref().unwrap().hits.iter().map(|(_, r)| r.pattern.clone()).collect()
        };
        assert_eq!(names(&b), want, "date, newest first, across directories");
        b.test_set_sort(SortKey::Date, false);
        frame(&mut b);
        let mut rev = want.to_vec();
        rev.reverse();
        assert_eq!(names(&b), rev, "date, oldest first");
        let _ = std::fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod drive_tests {
    use super::*;

    #[test]
    fn a_bare_drive_letter_is_its_root() {
        assert_eq!(drive_root(Path::new("C:")), PathBuf::from("C:/"));
        assert_eq!(drive_root(Path::new("d:")), PathBuf::from("d:/"));
        assert_eq!(drive_root(Path::new(" E: ")), PathBuf::from("E:/"));
        for keep in ["C:/", "C:\\", "C:/shots", "C:shots", "/Users", "1:", "", "ab"] {
            assert_eq!(drive_root(Path::new(keep)), PathBuf::from(keep), "{keep:?}");
        }
    }
}
