//! The project file: `name.scrub` (`.edify` before the rename, still
//! opened), pretty JSON, versioned.
//!
//! It holds the bin, the sequence (tracks, clips, speeds, dissolves, raster
//! and rate), the record and source marks, playheads and targeting, and for
//! every media file both its ABSOLUTE path and its path RELATIVE to the
//! project file, so a project moved with its media (or opened on another
//! machine with the same layout) still finds it. Media found by neither is
//! OFFLINE: its clips play black and the relink window offers to find it
//! (one file at a time, or a whole folder searched by file name).
//!
//! **A save never loses the last good file**: the document is written to a
//! temporary file beside it, flushed, then renamed over it in one step. An
//! AUTOSAVE goes to `~/Library/Application Support/Scrub/autosave.scrub`
//! while there are unsaved changes; a clean quit removes it, so finding one
//! at start-up means the last session ended badly, and recovery is offered.

use crate::editor::Editor;
use model::{Marks, Sequence, Targets};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
use std::time::{Instant, SystemTime};

/// The file format's version. A file from a newer Scrub is refused (in
/// words) rather than half read. 2: A2, A3 and M joined A1 (and the audio
/// target names a track); a version 1 file loads with them empty.
/// 3: text clips and named text styles; older files simply have none.
/// 4: Wobble clips (experimental); older files have none.
pub const VERSION: u32 = 4;
pub const EXT: &str = "scrub";
/// What projects were saved as before the rename; still opened.
pub const OLD_EXT: &str = "edify";
const RECENT_MAX: usize = 10;
/// How often unsaved work is autosaved.
const AUTOSAVE_SECS: u64 = 30;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BinEntry {
    pub path: PathBuf,
    #[serde(default)]
    pub rel: Option<PathBuf>,
    pub name: String,
    #[serde(default)]
    pub detail: String,
    /// The bin it is in ("" = clips; see `bin::Item::group`).
    #[serde(default)]
    pub group: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SourceState {
    pub path: PathBuf,
    #[serde(default)]
    pub rel: Option<PathBuf>,
    pub marks: Marks,
    pub playhead: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RecordState {
    pub marks: Marks,
    pub playhead: u64,
    pub targets: Targets,
    pub ripple: bool,
}

/// An optimised copy known when the project was saved (informational:
/// opening a file looks the copy up again, so a purged one is not trusted).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct OptimisedEntry {
    pub source: PathBuf,
    pub copy: PathBuf,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ProjectFile {
    /// The format version (the key was "edify" before the rename).
    #[serde(alias = "edify")]
    pub scrub: u32,
    pub sequence: Sequence,
    /// `media_rel[i]`: sequence media `i`'s path relative to the project.
    #[serde(default)]
    pub media_rel: Vec<Option<PathBuf>>,
    #[serde(default)]
    pub bin: Vec<BinEntry>,
    pub record: RecordState,
    #[serde(default)]
    pub source: Option<SourceState>,
    #[serde(default)]
    pub optimised: Vec<OptimisedEntry>,
    /// Set in an autosave: the project it is unsaved work of.
    #[serde(default)]
    pub autosave_of: Option<PathBuf>,
}

/// `path` relative to `base` (a folder), when they share a root.
pub fn relative(path: &Path, base: &Path) -> Option<PathBuf> {
    let p: Vec<Component> = path.components().collect();
    let b: Vec<Component> = base.components().collect();
    let common = p.iter().zip(&b).take_while(|(x, y)| x == y).count();
    if common == 0 {
        return None;
    }
    let mut out = PathBuf::new();
    for _ in common..b.len() {
        out.push("..");
    }
    for c in &p[common..] {
        out.push(c.as_os_str());
    }
    Some(out)
}

/// Write `bytes` to `path` without ever leaving a half-written file there:
/// a temporary file beside it, flushed to disk, renamed over it.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.saving"));
    let r = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        std::fs::rename(&tmp, path)
    })();
    if r.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    r
}

fn support_dir() -> PathBuf {
    media::sys::support_dir()
}

pub fn autosave_path() -> PathBuf {
    support_dir().join("autosave.scrub")
}

fn recent_path() -> PathBuf {
    support_dir().join("recent.json")
}

pub fn load_recent() -> Vec<PathBuf> {
    std::fs::read(recent_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

pub fn push_recent(p: &Path) {
    let mut v = load_recent();
    v.retain(|x| x != p);
    v.insert(0, p.to_path_buf());
    v.truncate(RECENT_MAX);
    let _ = std::fs::create_dir_all(support_dir());
    if let Ok(b) = serde_json::to_vec_pretty(&v) {
        let _ = write_atomic(&recent_path(), &b);
    }
}

/// Read and check a project file. Media paths are resolved (absolute, else
/// relative to the project); `missing` lists sequence media found by
/// neither.
pub fn read(path: &Path) -> Result<(ProjectFile, Vec<usize>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let head: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| format!("{} is not a project file: {e}", path.display()))?;
    let v = head.get("scrub").or_else(|| head.get("edify")).and_then(|v| v.as_u64()).ok_or_else(|| format!("{} is not a Scrub (or Edify) project", path.display()))?;
    if v as u32 > VERSION {
        return Err(format!("{} was saved by a newer Scrub (format {v}; this one reads {VERSION})", path.display()));
    }
    let mut p: ProjectFile = serde_json::from_value(head).map_err(|e| format!("{}: {e}", path.display()))?;
    p.sequence.after_load().map_err(|e| format!("{}: {e}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let find = |abs: &Path, rel: Option<&PathBuf>| -> Option<PathBuf> {
        if abs.exists() {
            return Some(abs.to_path_buf());
        }
        rel.map(|r| base.join(r)).filter(|p| p.exists())
    };
    let mut missing = Vec::new();
    for i in 0..p.sequence.media.len() {
        if p.sequence.media[i].generated {
            continue; // text: no file
        }
        let rel = p.media_rel.get(i).and_then(|r| r.as_ref());
        match find(&p.sequence.media[i].path, rel) {
            Some(f) => p.sequence.media[i].path = f,
            None => missing.push(i),
        }
    }
    for b in &mut p.bin {
        if let Some(f) = find(&b.path, b.rel.as_ref()) {
            b.path = f;
        }
    }
    if let Some(s) = &mut p.source {
        if let Some(f) = find(&s.path, s.rel.as_ref()) {
            s.path = f;
        }
    }
    Ok((p, missing))
}

/// Look for `name` under `dir`, a few folders deep.
pub fn search(dir: &Path, name: &std::ffi::OsStr, depth: usize) -> Option<PathBuf> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut dirs = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.file_name() == Some(name) && p.is_file() {
            return Some(p);
        }
        if depth > 0 && p.is_dir() && !e.file_name().to_string_lossy().starts_with('.') {
            dirs.push(p);
        }
    }
    dirs.into_iter().find_map(|d| search(&d, name, depth - 1))
}

/// The project side of the editor: file, dirty state, autosave, recovery,
/// relinking, the quit question.
pub struct ProjectState {
    pub path: Option<PathBuf>,
    /// The document as last saved or opened (dirty = it differs).
    saved: Option<String>,
    pub dirty: bool,
    checked: Instant,
    autosaved: Instant,
    /// Sequence media indices that could not be found.
    pub missing: Vec<usize>,
    pub relink_dir: String,
    /// An autosave found at start-up: (file, when written).
    pub recover: Option<(PathBuf, SystemTime)>,
    /// Asking whether to save before quitting.
    pub asking_quit: bool,
    pub recent: Vec<PathBuf>,
    /// Tests and headless checks: no autosave, no recovery, no prompts.
    pub headless: bool,
}

impl ProjectState {
    pub fn new(headless: bool) -> Self {
        let recover = if headless {
            None
        } else {
            let p = autosave_path();
            std::fs::metadata(&p).ok().and_then(|m| m.modified().ok()).map(|t| (p, t))
        };
        Self {
            path: None,
            saved: None,
            dirty: false,
            checked: Instant::now(),
            autosaved: Instant::now(),
            missing: Vec::new(),
            relink_dir: String::new(),
            recover,
            asking_quit: false,
            recent: if headless { Vec::new() } else { load_recent() },
            headless,
        }
    }

    pub fn name(&self) -> String {
        self.path.as_ref().and_then(|p| p.file_stem()).map_or("untitled".into(), |n| n.to_string_lossy().into_owned())
    }
}

impl Editor {
    /// The document as it would be saved to `path` (relative paths from it).
    pub fn project_doc(&self, path: Option<&Path>) -> ProjectFile {
        let base = path.and_then(|p| p.parent());
        let rel = |p: &Path| base.and_then(|b| relative(p, b));
        let seq = self.rec.seq.clone();
        let media_rel = seq.media.iter().map(|m| if m.generated { None } else { rel(&m.path) }).collect();
        let bin = self
            .bin
            .items
            .iter()
            .map(|it| BinEntry { path: it.path.clone(), rel: rel(&it.path), name: it.name.clone(), detail: it.detail.clone(), group: it.group.clone() })
            .collect();
        let source = self.source.as_ref().map(|c| SourceState {
            path: c.original.clone(),
            rel: rel(&c.original),
            marks: c.marks,
            playhead: c.transport.frame(),
        });
        let optimised = seq
            .media
            .iter()
            .filter_map(|m| self.rec.program.read_path(&m.path).filter(|r| *r != m.path).map(|r| OptimisedEntry { source: m.path.clone(), copy: r.to_path_buf() }))
            .collect();
        ProjectFile {
            scrub: VERSION,
            sequence: seq,
            media_rel,
            bin,
            record: RecordState {
                marks: self.rec.marks,
                playhead: self.rec.transport.frame(),
                targets: self.rec.targets,
                ripple: self.rec.ripple,
            },
            source,
            optimised,
            autosave_of: None,
        }
    }

    fn doc_text(&self, path: Option<&Path>) -> String {
        serde_json::to_string_pretty(&self.project_doc(path)).unwrap_or_default()
    }

    /// Save to `path` (temp + rename). The project becomes that file.
    pub fn save_to(&mut self, path: &Path) -> Result<(), String> {
        // (An old `.edify` project saves where it is, as itself.)
        let path = if path.extension().is_some_and(|e| e == EXT || e == OLD_EXT) { path.to_path_buf() } else { path.with_extension(EXT) };
        let text = self.doc_text(Some(&path));
        write_atomic(&path, text.as_bytes()).map_err(|e| format!("saving {}: {e}", path.display()))?;
        self.project.path = Some(path.clone());
        self.project.saved = Some(text);
        self.project.dirty = false;
        self.post = crate::marketing::PostCtx::for_project(&path);
        if !self.project.headless {
            push_recent(&path);
            self.project.recent = load_recent();
            let _ = std::fs::remove_file(autosave_path());
        }
        self.say(format!("saved {}", path.display()));
        Ok(())
    }

    /// cmd+S: save, or ask where for an untitled project.
    pub fn save(&mut self, ask: bool) {
        // EXPERIMENT (fold): the open wob comp (its knob values) with it.
        #[cfg(feature = "fold")]
        self.fold_save_quiet();
        let path = match (&self.project.path, ask) {
            (Some(p), false) => Some(p.clone()),
            _ if self.project.headless => None,
            _ => {
                // Untitled media in an organisicator project: its shot's
                // Scrub folder and a versioned name.
                let suggest = match &self.project.path {
                    Some(_) => None,
                    None => self.first_media().and_then(|m| crate::organisicator::save_suggestion(&m)),
                };
                let name = suggest.as_ref().and_then(|s| s.file_name()).map_or(format!("{}.{EXT}", self.project.name()), |n| n.to_string_lossy().into_owned());
                let mut d = imfram::filedialog::FileDialog::new().add_filter("Scrub project", &[EXT, OLD_EXT]).set_file_name(name);
                if let Some(dir) = self.project.path.as_ref().and_then(|p| p.parent()).or(suggest.as_ref().and_then(|s| s.parent())) {
                    if let Some(sd) = suggest.as_ref().and_then(|s| s.parent()) {
                        let _ = std::fs::create_dir_all(sd);
                    }
                    d = d.set_directory(dir);
                }
                d.save_file()
            }
        };
        if let Some(p) = path {
            if let Err(e) = self.save_to(&p) {
                self.say(e);
            }
        }
    }

    /// The first media this project uses (sequence, else source, else bin).
    pub fn first_media(&self) -> Option<PathBuf> {
        self.rec
            .seq
            .media
            .first()
            .map(|m| m.path.clone())
            .or_else(|| self.source.as_ref().map(|c| c.original.clone()))
            .or_else(|| self.bin.items.first().map(|i| i.path.clone()))
    }

    /// cmd+O: open a project from a file dialog.
    pub fn open_project_dialog(&mut self) {
        if self.project.headless {
            return;
        }
        let mut d = imfram::filedialog::FileDialog::new().add_filter("Scrub project", &[EXT, OLD_EXT]);
        if let Some(dir) = self.project.path.as_ref().and_then(|p| p.parent()) {
            d = d.set_directory(dir);
        }
        if let Some(p) = d.pick_file() {
            self.open_project(&p);
        }
    }

    /// Replace the session with a project file.
    pub fn open_project(&mut self, path: &Path) {
        match read(path) {
            Ok((doc, missing)) => {
                let autosave = doc.autosave_of.clone();
                self.apply(doc);
                self.project.missing = missing;
                if path == autosave_path() {
                    // Recovered work is unsaved by definition.
                    self.project.path = autosave;
                    self.project.saved = None;
                    self.project.dirty = true;
                } else {
                    self.project.path = Some(path.to_path_buf());
                    self.project.saved = Some(self.doc_text(Some(path)));
                    self.project.dirty = false;
                    if !self.project.headless {
                        push_recent(path);
                        self.project.recent = load_recent();
                    }
                }
                self.post = crate::marketing::PostCtx::for_project(self.project.path.as_deref().unwrap_or(path));
                let n = self.project.missing.len();
                if n > 0 {
                    self.say(format!("{n} media file(s) offline: file > relink media…"));
                } else {
                    self.say(format!("opened {}", path.display()));
                }
            }
            Err(e) => self.say(e),
        }
    }

    /// Start a fresh project (a new post's edit): an empty bin and source,
    /// this sequence, to be saved at `path` (unsaved until then).
    pub fn new_project(&mut self, seq: Sequence, path: Option<PathBuf>) {
        let r = &mut self.rec;
        r.seq = seq;
        r.history.reset(&r.seq);
        r.program.reset();
        r.marks = Marks::default();
        r.sel = crate::timelineui::Selection::None;
        r.transport.stop();
        r.transport.frames = 1;
        r.transport.seek(0);
        r.shown = None;
        r.view.fit();
        self.bin.items.clear();
        self.bin.sel = 0;
        self.source = None;
        self.project.path = path;
        self.project.saved = None;
        self.project.missing.clear();
        self.post = None;
    }

    fn apply(&mut self, doc: ProjectFile) {
        let r = &mut self.rec;
        r.seq = doc.sequence;
        r.history.reset(&r.seq);
        r.program.reset();
        r.marks = doc.record.marks;
        r.targets = doc.record.targets;
        // The trim mode is not restored: every session starts on plain
        // trims (B for ripple), so a saved ripple never surprises a drag.
        let _ = doc.record.ripple;
        r.ripple = false;
        r.sel = crate::timelineui::Selection::None;
        r.transport.stop();
        r.fit_transport();
        r.transport.seek(doc.record.playhead);
        r.shown = None;
        r.view.fit();
        self.bin.items = doc
            .bin
            .into_iter()
            .map(|b| crate::bin::Item { path: b.path, name: b.name, detail: b.detail, group: b.group })
            .collect();
        self.bin.sel = 0;
        self.source = None;
        if let Some(s) = doc.source {
            if s.path.exists() {
                self.open(&s.path);
                if let Some(c) = &mut self.source {
                    c.marks = s.marks;
                    c.transport.seek(s.playhead);
                }
            }
        }
    }

    /// Every half second: is the document different from what was saved?
    /// Every `AUTOSAVE_SECS` with unsaved changes: autosave.
    pub fn project_tick(&mut self) {
        if self.project.checked.elapsed().as_millis() < 500 {
            return;
        }
        self.project.checked = Instant::now();
        let path = self.project.path.clone();
        let text = self.doc_text(path.as_deref());
        let empty = self.rec.seq.is_empty() && self.bin.items.is_empty();
        self.project.dirty = match &self.project.saved {
            Some(s) => *s != text,
            None => !empty,
        };
        if self.project.dirty && !self.project.headless && self.project.autosaved.elapsed().as_secs() >= AUTOSAVE_SECS {
            self.project.autosaved = Instant::now();
            let mut doc = self.project_doc(path.as_deref());
            doc.autosave_of = path;
            if let Ok(t) = serde_json::to_string_pretty(&doc) {
                let _ = std::fs::create_dir_all(support_dir());
                let _ = write_atomic(&autosave_path(), t.as_bytes());
            }
        }
    }

    /// A clean exit: the autosave is not needed any more.
    pub fn project_clean_exit(&mut self) {
        if !self.project.headless {
            let _ = std::fs::remove_file(autosave_path());
        }
    }

    /// Point offline media `i` at `path` (its clips then play).
    pub fn relink(&mut self, i: usize, path: PathBuf) {
        if let Some(m) = self.rec.seq.media.get_mut(i) {
            m.path = path;
        }
        self.project.missing.retain(|&x| x != i && self.rec.seq.media.get(x).is_some_and(|m| !m.path.exists()));
        self.rec.program.reset();
        self.rec.history.reset(&self.rec.seq);
    }

    /// Look in a folder (a few levels down) for every offline file by name.
    pub fn relink_search(&mut self, dir: &Path) -> usize {
        let mut found = 0;
        for i in self.project.missing.clone() {
            let Some(name) = self.rec.seq.media[i].path.file_name().map(|n| n.to_os_string()) else { continue };
            if let Some(p) = search(dir, &name, 4) {
                self.relink(i, p);
                found += 1;
            }
        }
        found
    }

    /// The project windows: recovery, relink, the quit question. True when
    /// the app may quit now.
    pub fn project_windows(&mut self, ctx: &egui::Context) -> bool {
        let mut quit = false;
        if let Some((p, when)) = self.project.recover.clone() {
            let ago = when.elapsed().map(|d| d.as_secs() / 60).unwrap_or(0);
            egui::Window::new("recover unsaved work").collapsible(false).resizable(false).show(ctx, |ui| {
                ui.label(format!("Scrub did not quit cleanly last time. Unsaved work from {ago} min ago was kept."));
                ui.horizontal(|ui| {
                    if ui.button("recover").clicked() {
                        self.open_project(&p);
                        self.project.recover = None;
                    }
                    if ui.button("discard").clicked() {
                        // Kept aside, not deleted: it is still the user's work.
                        let _ = std::fs::rename(&p, p.with_extension("scrub.discarded"));
                        self.project.recover = None;
                    }
                });
            });
        }
        if !self.project.missing.is_empty() && self.relink_open {
            let mut open = true;
            egui::Window::new("relink media").open(&mut open).default_width(520.0).show(ctx, |ui| {
                ui.label("These files are not where the project says, nor beside it:");
                for i in self.project.missing.clone() {
                    let p = self.rec.seq.media[i].path.clone();
                    ui.horizontal(|ui| {
                        ui.label(p.display().to_string());
                        if ui.small_button("locate…").clicked() {
                            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                            if let Some(f) = imfram::filedialog::FileDialog::new().set_file_name(name).pick_file() {
                                self.relink(i, f);
                            }
                        }
                    });
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("search a folder");
                    ui.text_edit_singleline(&mut self.project.relink_dir);
                    if ui.button("choose…").clicked() {
                        if let Some(d) = imfram::filedialog::FileDialog::new().pick_folder() {
                            self.project.relink_dir = d.display().to_string();
                        }
                    }
                    if ui.button("search").clicked() {
                        let d = imfram::filebrowser::expand_home(&self.project.relink_dir);
                        let n = self.relink_search(&d);
                        self.say(format!("relinked {n} file(s)"));
                    }
                });
            });
            self.relink_open = open;
        }
        if self.project.asking_quit {
            egui::Window::new("unsaved changes").collapsible(false).resizable(false).show(ctx, |ui| {
                ui.label(format!("Save changes to {} before quitting?", self.project.name()));
                ui.horizontal(|ui| {
                    if ui.button("save").clicked() {
                        self.save(false);
                        if !self.project.dirty {
                            quit = true;
                        }
                        self.project.asking_quit = false;
                    }
                    if ui.button("don't save").clicked() {
                        quit = true;
                        self.project.asking_quit = false;
                    }
                    if ui.button("cancel").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        self.project.asking_quit = false;
                    }
                });
            });
        }
        quit
    }

    /// Quit asked (cmd+Q, the window's close): straight away when saved,
    /// else ask. True when it may go now.
    pub fn request_quit(&mut self) -> bool {
        self.project.checked = Instant::now() - std::time::Duration::from_secs(1);
        self.project_tick();
        if self.project.dirty && !self.project.headless {
            self.project.asking_quit = true;
            false
        } else {
            true
        }
    }

    /// The window title: project name, a dot while unsaved.
    pub fn title(&self) -> String {
        format!("{}{} — Scrub", self.project.name(), if self.project.dirty { " •" } else { "" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_walk_up_and_down() {
        assert_eq!(relative(Path::new("/a/b/media/c.mov"), Path::new("/a/b/proj")), Some(PathBuf::from("../media/c.mov")));
        assert_eq!(relative(Path::new("/a/b/c.mov"), Path::new("/a/b")), Some(PathBuf::from("c.mov")));
    }

    #[cfg(unix)]
    #[test]
    fn a_failed_save_leaves_the_last_good_file() {
        let dir = std::env::temp_dir().join(format!("scrub-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("p.edify");
        write_atomic(&p, b"good").unwrap();
        // A save that cannot even make its temporary file fails...
        let bad = dir.join("no-such-folder").join("p.edify");
        assert!(write_atomic(&bad, b"new").is_err());
        // ...and one into a folder made read-only fails without touching it.
        let mut perms = std::fs::metadata(&dir).unwrap().permissions();
        use std::os::unix::fs::PermissionsExt;
        perms.set_mode(0o555);
        std::fs::set_permissions(&dir, perms.clone()).unwrap();
        let r = write_atomic(&p, b"new");
        perms.set_mode(0o755);
        std::fs::set_permissions(&dir, perms).unwrap();
        assert!(r.is_err());
        assert_eq!(std::fs::read(&p).unwrap(), b"good");
        assert!(!dir.join(".p.edify.saving").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn doc(media: &Path) -> ProjectFile {
        let mut s = Sequence::new(model::Fps::new(24, 1));
        let m = s.add_media(model::Media {
            path: media.to_path_buf(),
            name: "c.mov".into(),
            frames: 100,
            fps: model::Fps::new(24, 1),
            has_audio: false,
            width: 64,
            height: 36,
            generated: false,
        });
        s.overwrite(model::Source::new(m, 0, 10), 0, Targets::default());
        ProjectFile {
            scrub: VERSION,
            sequence: s,
            media_rel: vec![Some(PathBuf::from("media/c.mov"))],
            bin: vec![],
            record: RecordState { marks: Marks { inp: Some(2), out: None }, playhead: 5, targets: Targets::default(), ripple: true },
            source: None,
            optimised: vec![],
            autosave_of: None,
        }
    }

    #[test]
    fn a_project_round_trips_and_finds_moved_media_by_its_relative_path() {
        let dir = std::env::temp_dir().join(format!("scrub-proj-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("media")).unwrap();
        let media = dir.join("media/c.mov");
        std::fs::write(&media, b"x").unwrap();
        // Saved with an absolute path that no longer exists (the project
        // folder moved): the relative path finds it.
        let d = doc(Path::new("/nowhere/c.mov"));
        let p = dir.join("p.edify");
        write_atomic(&p, serde_json::to_string_pretty(&d).unwrap().as_bytes()).unwrap();
        let (back, missing) = read(&p).unwrap();
        assert!(missing.is_empty());
        assert_eq!(back.sequence.media[0].path, dir.join("media/c.mov"));
        assert_eq!(back.record, d.record);
        assert_eq!(back.sequence.tracks, d.sequence.tracks);
        // Gone from both: offline, and the rest still loads.
        std::fs::remove_file(&media).unwrap();
        let (_, missing) = read(&p).unwrap();
        assert_eq!(missing, vec![0]);
        assert_eq!(search(&dir, std::ffi::OsStr::new("p.edify"), 2), Some(p.clone()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_newer_format_is_refused_in_words() {
        let dir = std::env::temp_dir().join(format!("scrub-newer-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("n.edify");
        std::fs::write(&p, br#"{"edify": 99}"#).unwrap();
        assert!(read(&p).unwrap_err().contains("newer Scrub"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_version_1_project_loads_with_the_new_audio_tracks() {
        let dir = std::env::temp_dir().join(format!("scrub-v1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let media = dir.join("c.mov");
        std::fs::write(&media, b"x").unwrap();
        let mut v = serde_json::to_value(doc(&media)).unwrap();
        // (An Edify-era file: the version under its old key.)
        v.as_object_mut().unwrap().remove("scrub");
        v["edify"] = 1.into();
        v["sequence"]["tracks"].as_array_mut().unwrap().truncate(4);
        for t in v["sequence"]["tracks"].as_array_mut().unwrap() {
            for k in ["mute", "solo", "gain_db"] {
                t.as_object_mut().unwrap().remove(k);
            }
        }
        v["record"]["targets"]["audio"] = true.into();
        let p = dir.join("old.edify");
        std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
        let (back, _) = read(&p).unwrap();
        assert_eq!(back.sequence.tracks.len(), 7);
        assert_eq!(back.sequence.tracks[model::V1].clips.len(), 1);
        assert_eq!(back.record.targets.audio, Some(model::A1));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
