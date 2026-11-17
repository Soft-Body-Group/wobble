//! Managed projects, read from the structure each one publishes inside
//! itself: `<project>/.organisicator/project.json`, written by the
//! organisicator project manager. Every path in it is relative to the
//! project root, so the same file holds on macOS, Windows and Linux,
//! wherever the shared drive mounts the managed root. Nothing here needs
//! the manager running.
//!
//! This is what lets wobble work in a project without ever showing a file
//! system: a Load names its footage by project, shot and place under the
//! shot (`assets::AssetRef`), a comp is opened from and saved to a shot's
//! comp folder, and each machine resolves those names against its OWN
//! managed root. The comp-folder rule is the one the organisicator add-on
//! (`plugins/organisicator/shots.py`) uses, so the two never disagree.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const MANIFEST_DIR: &str = ".organisicator";
pub const MANIFEST_FILE: &str = "project.json";
const SUPPORTED_FORMAT: u64 = 1;
pub const COMP_EXT: &str = "wbl";

/// One anatomy folder: a path with `{variable}` segments for repeating levels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Folder {
    pub path: String,
    pub kind: String,
    pub exts: String,
}

/// A concrete folder behind a repeating level (a sequence, a shot…).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inst {
    pub id: i64,
    pub parent: Option<i64>,
    pub decl: String,
    pub value: String,
    pub rel: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Project {
    pub name: String,
    pub code: String,
    /// On THIS machine.
    pub root: PathBuf,
    pub folders: Vec<Folder>,
    pub instances: Vec<Inst>,
}

fn segs(rel: &str) -> Vec<&str> {
    rel.split(['/', '\\']).filter(|s| !s.is_empty()).collect()
}

fn is_var(seg: &str) -> bool {
    seg.len() > 2
        && seg.starts_with('{')
        && seg.ends_with('}')
        && seg[1..seg.len() - 1].chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `rel` ('/'-separated, from any OS) under `base`, with this OS's separators.
pub fn join_rel(base: &Path, rel: &str) -> PathBuf {
    let mut p = base.to_path_buf();
    for s in segs(rel) {
        p.push(s);
    }
    p
}

impl Project {
    pub fn instance(&self, id: i64) -> Option<&Inst> {
        self.instances.iter().find(|i| i.id == id)
    }

    /// The levels under `parent` (None = the project), sorted by name.
    pub fn children(&self, parent: Option<i64>) -> Vec<&Inst> {
        let mut v: Vec<&Inst> = self.instances.iter().filter(|i| i.parent == parent).collect();
        v.sort_by_key(|i| i.value.to_lowercase());
        v
    }

    /// Enclosing levels, outermost first, then `inst` itself.
    pub fn chain<'a>(&'a self, inst: &'a Inst) -> Vec<&'a Inst> {
        let mut out = vec![inst];
        let mut p = inst.parent.and_then(|id| self.instance(id));
        while let Some(i) = p.filter(|_| out.len() < 32) {
            out.push(i);
            p = i.parent.and_then(|id| self.instance(id));
        }
        out.reverse();
        out
    }

    /// The names a reference stores for `inst`: sequence, shot…
    pub fn names(&self, inst: Option<&Inst>) -> Vec<String> {
        inst.map(|i| self.chain(i).iter().map(|c| c.value.clone()).collect()).unwrap_or_default()
    }

    /// The level these names lead to; empty names = the project itself.
    pub fn by_names(&self, names: &[String]) -> Option<Option<&Inst>> {
        let mut at: Option<&Inst> = None;
        for n in names {
            at = Some(*self.children(at.map(|i| i.id)).iter().find(|c| c.value == *n)?);
        }
        Some(at)
    }

    pub fn path_of(&self, inst: Option<&Inst>) -> PathBuf {
        match inst {
            None => self.root.clone(),
            Some(i) => join_rel(&self.root, &i.rel),
        }
    }

    /// The anatomy folders that sit directly inside `inst` (not inside a
    /// deeper repeating level), with where each is on this machine.
    fn folders_of(&self, inst: Option<&Inst>) -> Vec<(&Folder, PathBuf)> {
        let mut out = Vec::new();
        for f in &self.folders {
            let s = segs(&f.path);
            let vars: Vec<usize> = (0..s.len()).filter(|&k| is_var(s[k])).collect();
            match inst {
                None if vars.is_empty() => out.push((f, join_rel(&self.root, &s.join("/")))),
                None => {}
                Some(i) => {
                    let Some(&k) = vars.last() else { continue };
                    if s[..=k].join("/") != segs(&i.decl).join("/") || k + 1 >= s.len() {
                        continue;
                    }
                    out.push((f, join_rel(&join_rel(&self.root, &i.rel), &s[k + 1..].join("/"))));
                }
            }
        }
        out
    }

    /// The best-scoring folder of `inst`; if it has none, of the nearest
    /// enclosing level that does, then the project's own.
    fn pick(&self, inst: Option<&Inst>, score: impl Fn(&Folder) -> u32) -> Option<PathBuf> {
        let mut levels: Vec<Option<&Inst>> = Vec::new();
        if let Some(i) = inst {
            levels.extend(self.chain(i).into_iter().rev().map(Some));
        }
        levels.push(None);
        for l in levels {
            let best = self
                .folders_of(l)
                .into_iter()
                .map(|(f, p)| (score(f), p))
                .filter(|(s, _)| *s > 0)
                .max_by_key(|(s, _)| *s);
            if let Some((_, p)) = best {
                return Some(p);
            }
        }
        None
    }

    /// Where `inst`'s comps live: a folder named wobble, else one whose
    /// file types include .wbl, else a comp folder, else its software
    /// folder (or that folder's `wobble` child when one exists). With no
    /// anatomy to say, a `comp` folder in the shot (where a collect puts it).
    pub fn comp_dir(&self, inst: Option<&Inst>) -> PathBuf {
        let found = self.pick(inst, |f| {
            let name = segs(&f.path).last().map(|s| s.to_lowercase()).unwrap_or_default();
            let kind = f.kind.to_lowercase();
            if name == "wobble" {
                5
            } else if f.exts.to_lowercase().split_whitespace().any(|e| e == COMP_EXT) {
                4
            } else if name == "comp" || name == "comps" || kind == "comp" {
                3
            } else if kind == "software" || ["software", "scripts", "work"].contains(&name.as_str()) {
                2
            } else {
                0
            }
        });
        match found {
            Some(d) if d.file_name().is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case("wobble")) => d,
            Some(d) => ["wobble", "Wobble"].iter().map(|c| d.join(c)).find(|p| p.is_dir()).unwrap_or(d),
            None => self.path_of(inst).join("comp"),
        }
    }

    /// Where `inst`'s plates go: a plate(s) folder by type, then by name,
    /// then a source folder; with none, a `plates` folder in the shot.
    pub fn plates_dir(&self, inst: Option<&Inst>) -> PathBuf {
        self.pick(inst, |f| {
            let name = segs(&f.path).last().map(|s| s.to_lowercase()).unwrap_or_default();
            let kind = f.kind.to_lowercase();
            if kind == "plate" || kind == "plates" {
                4
            } else if name == "plates" || name == "plate" {
                3
            } else if kind == "source" {
                2
            } else {
                0
            }
        })
        .unwrap_or_else(|| self.path_of(inst).join("plates"))
    }

    /// The folders of other levels inside `inst`'s: they belong to them.
    fn nested(&self, inst: Option<&Inst>) -> Vec<PathBuf> {
        let me = self.path_of(inst);
        self.instances
            .iter()
            .filter(|i| Some(i.id) != inst.map(|x| x.id))
            .map(|i| self.path_of(Some(i)))
            .filter(|d| d.starts_with(&me) && *d != me)
            .collect()
    }

    /// The deepest level whose folder holds `path`, and `path` under it
    /// ('/'-separated). None when `path` is not in this project.
    pub fn locate(&self, path: &Path) -> Option<(Option<&Inst>, String)> {
        path.strip_prefix(&self.root).ok()?;
        let mut best: Option<&Inst> = None;
        for i in &self.instances {
            if path.starts_with(self.path_of(Some(i)))
                && best.is_none_or(|b| segs(&i.rel).len() > segs(&b.rel).len())
            {
                best = Some(i);
            }
        }
        let rel = path.strip_prefix(self.path_of(best)).ok()?;
        Some((best, rel.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/")))
    }
}

pub fn load_manifest(project_root: &Path) -> Option<Project> {
    let text = std::fs::read_to_string(project_root.join(MANIFEST_DIR).join(MANIFEST_FILE)).ok()?;
    parse_manifest(&text, project_root)
}

pub fn parse_manifest(text: &str, project_root: &Path) -> Option<Project> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    if v["app"].as_str() != Some("organisicator") || v["format"].as_u64().unwrap_or(0) > SUPPORTED_FORMAT {
        return None;
    }
    let s = |x: &serde_json::Value| x.as_str().unwrap_or_default().to_string();
    let folders = v["folders"]
        .as_array()
        .map(|a| a.iter().map(|f| Folder { path: s(&f["path"]), kind: s(&f["type"]), exts: s(&f["exts"]) }).collect())
        .unwrap_or_default();
    let instances = v["instances"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|i| {
                    Some(Inst {
                        id: i["id"].as_i64()?,
                        parent: i["parent"].as_i64(),
                        decl: i["decl"].as_str()?.to_string(),
                        value: i["value"].as_str()?.to_string(),
                        rel: i["rel"].as_str()?.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let name = v["name"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| project_root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    Some(Project { name, code: s(&v["code"]), root: project_root.to_path_buf(), folders, instances })
}

/// Every project under `managed` (a project's own subtree is not searched
/// further). Hidden and `_` folders are skipped, as the add-on skips them.
pub fn find_projects(managed: &Path) -> Vec<Project> {
    fn walk(d: &Path, depth: u32, out: &mut Vec<Project>) {
        if d.join(MANIFEST_DIR).join(MANIFEST_FILE).is_file() {
            if let Some(p) = load_manifest(d) {
                out.push(p);
                return;
            }
        }
        if depth >= 6 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(d) else { return };
        let mut dirs: Vec<PathBuf> = rd
            .flatten()
            .filter(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                !n.starts_with('.') && !n.starts_with('_') && e.file_type().is_ok_and(|t| t.is_dir())
            })
            .map(|e| e.path())
            .collect();
        dirs.sort_by_key(|p| p.to_string_lossy().to_lowercase());
        for p in dirs {
            walk(&p, depth + 1, out);
        }
    }
    let mut out = Vec::new();
    if managed.is_dir() {
        walk(managed, 0, &mut out);
    }
    out.sort_by_key(|p| p.name.to_lowercase());
    out
}

// ---- the managed root -------------------------------------------------------

/// Shared with the organisicator add-on, so setting it in either place sets both.
fn prefs_file() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(PathBuf::from(home).join(".wobble").join("organisicator.json"))
}

/// Where the manager puts projects when nothing says otherwise.
const DEFAULT_ROOTS: &[&str] = &["~/Library/CloudStorage/Dropbox/managed", "~/Dropbox/managed", "D:/Dropbox/managed"];

/// This machine's managed root: `$WOBBLE_PROJECTS_ROOT` (suites), the
/// remembered one, else the first default that exists.
pub fn managed_root() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("WOBBLE_PROJECTS_ROOT") {
        return Some(PathBuf::from(p));
    }
    let saved = prefs_file()
        .and_then(|f| std::fs::read_to_string(f).ok())
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["root"].as_str().map(str::to_string))
        .filter(|s| !s.trim().is_empty());
    if let Some(s) = saved {
        return Some(engine::paths::expand_home(&s));
    }
    DEFAULT_ROOTS.iter().map(|r| engine::paths::expand_home(r)).find(|p| p.is_dir())
}

pub fn set_managed_root(root: &str) -> Result<(), String> {
    let f = prefs_file().ok_or("no home folder")?;
    let mut v: serde_json::Value = std::fs::read_to_string(&f)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .filter(|v: &serde_json::Value| v.is_object())
        .unwrap_or_else(|| serde_json::json!({}));
    v["root"] = serde_json::Value::String(root.trim().to_string());
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    std::fs::write(&f, serde_json::to_string_pretty(&v).unwrap_or_default()).map_err(|e| e.to_string())?;
    forget();
    Ok(())
}

/// The projects under the managed root, scanned once and kept until a
/// refresh (`forget`) or a lookup that misses.
static CACHE: Mutex<Option<(PathBuf, Vec<Project>)>> = Mutex::new(None);

pub fn projects(rescan: bool) -> Vec<Project> {
    let Some(root) = managed_root() else { return Vec::new() };
    let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if rescan || c.as_ref().is_none_or(|(r, _)| *r != root) {
        *c = Some((root.clone(), find_projects(&root)));
    }
    c.as_ref().map(|(_, p)| p.clone()).unwrap_or_default()
}

pub fn forget() {
    *CACHE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

// ---- comps ----------------------------------------------------------------------

/// `.wbl` files in a folder, newest first.
pub fn comps(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_lowercase();
            !n.starts_with('.') && n.ends_with(".wbl") && e.file_type().is_ok_and(|t| t.is_file())
        })
        .map(|e| (e.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH), e.path()))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    v.into_iter().map(|(_, p)| p).collect()
}

/// `cf_010_light_v003` → `cf_010_light_v004`; no version → `_v002`.
pub fn next_version(stem: &str) -> String {
    let b = stem.as_bytes();
    let mut d = b.len();
    while d > 0 && b[d - 1].is_ascii_digit() {
        d -= 1;
    }
    if d < b.len() && d > 0 && (b[d - 1] == b'v' || b[d - 1] == b'V') {
        let n: u64 = stem[d..].parse().unwrap_or(0);
        return format!("{}{:0w$}", &stem[..d], n + 1, w = b.len() - d);
    }
    format!("{stem}_v002")
}

/// A name that is safe as a file name on every OS.
fn safe(s: &str) -> String {
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join("_")
        .chars()
        .filter(|c| !"<>:\"|?*/\\".contains(*c))
        .collect()
}

/// `<shot>_<task>_v001.wbl`, or the next version past any already in `dir`.
pub fn new_comp_name(p: &Project, inst: Option<&Inst>, task: &str, dir: &Path) -> String {
    let base = safe(inst.map_or(&p.name, |i| &i.value));
    let task = safe(task);
    let mut stem = if task.is_empty() { format!("{base}_v001") } else { format!("{base}_{task}_v001") };
    while dir.join(format!("{stem}.{COMP_EXT}")).exists() {
        stem = next_version(&stem);
    }
    format!("{stem}.{COMP_EXT}")
}

/// `…_v004.wbl` → `…_v005.wbl` beside it, skipping versions already there.
pub fn version_up(path: &Path) -> PathBuf {
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    loop {
        stem = next_version(&stem);
        let p = dir.join(format!("{stem}.{COMP_EXT}"));
        if !p.exists() {
            return p;
        }
    }
}

// ---- media ----------------------------------------------------------------------

/// `name.0101.exr` → (`name.`, 4, `.exr`).
fn frame_parts(name: &str) -> Option<(&str, usize, &str)> {
    let dot = name.rfind('.')?;
    let (stem, ext) = name.split_at(dot);
    let digits = stem.bytes().rev().take_while(u8::is_ascii_digit).count();
    (digits > 0).then(|| (&stem[..stem.len() - digits], digits, ext))
}

/// What a Load can open under `inst`'s folder (not inside another level's),
/// as places under it: `plates/bg/bg.####.exr` for frames, the file for a
/// movie or a still. Sorted; hidden folders skipped.
pub fn media(p: &Project, inst: Option<&Inst>) -> Vec<String> {
    let base = p.path_of(inst);
    let nested = p.nested(inst);
    let mut out = Vec::new();
    fn walk(d: &Path, base: &Path, nested: &[PathBuf], left: u32, out: &mut Vec<String>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        let mut groups: std::collections::BTreeMap<(String, usize, String), usize> = Default::default();
        let mut files = Vec::new();
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let path = e.path();
            if e.file_type().is_ok_and(|t| t.is_dir()) {
                if left > 0 && !nested.contains(&path) {
                    walk(&path, base, nested, left - 1, out);
                }
                continue;
            }
            if !engine::decode::is_supported(&path) {
                continue;
            }
            let movie = matches!(
                path.extension().map(|x| x.to_string_lossy().to_lowercase()).as_deref(),
                Some("mov" | "mp4" | "mxf" | "mkv" | "avi" | "webm")
            );
            match frame_parts(&name).filter(|_| !movie) {
                Some((head, pad, ext)) => *groups.entry((head.into(), pad, ext.into())).or_default() += 1,
                None => files.push(name),
            }
        }
        let rel = |n: &str| {
            let full = d.join(n);
            full.strip_prefix(base)
                .map(|r| r.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join("/"))
                .unwrap_or_default()
        };
        for ((head, pad, ext), n) in groups {
            // One numbered file is a still that happens to end in digits.
            if n == 1 {
                if let Some(f) = std::fs::read_dir(d).ok().into_iter().flatten().flatten().find(|e| {
                    let s = e.file_name().to_string_lossy().into_owned();
                    frame_parts(&s).is_some_and(|(h, p, x)| h == head && p == pad && x == ext)
                }) {
                    out.push(rel(&f.file_name().to_string_lossy()));
                }
                continue;
            }
            out.push(rel(&format!("{head}{}{ext}", "#".repeat(pad))));
        }
        for f in files {
            out.push(rel(&f));
        }
    }
    walk(&base, &base, &nested, 4, &mut out);
    out.sort();
    out
}

// ---- bringing footage in -------------------------------------------------------

/// Copies a clip's files (`frames`: a sequence's frames, or one movie or
/// still) into a folder of its own under `plates`, named for the clip. A
/// file already there at the same size is kept; a DIFFERENT file of the
/// same name is never overwritten — the clip goes in `<name>_2` instead.
/// Returns the clip as a Load names it there (`…/bg.####.exr`).
/// Blocking: run it on a worker.
/// A clip's name for its folder (`bg` for `bg.####.exr` or `bg.mov`) and
/// its file name as a Load names it (`bg.####.exr`).
pub fn clip_names(frames: &[PathBuf]) -> Result<(String, String), String> {
    let mut files: Vec<&PathBuf> = frames.iter().collect();
    files.dedup();
    let first = files.first().ok_or("nothing to copy")?;
    let fname = first.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or("no file name")?;
    let parts = frame_parts(&fname).filter(|_| files.len() > 1);
    let (stem, pattern) = match parts {
        Some((head, pad, ext)) => (head.trim_end_matches(['.', '_', '-']).to_string(), format!("{head}{}{ext}", "#".repeat(pad))),
        None => (fname.rsplit_once('.').map_or(fname.as_str(), |(a, _)| a).to_string(), fname.clone()),
    };
    Ok((if safe(&stem).is_empty() { "clip".to_string() } else { safe(&stem) }, pattern))
}

pub fn copy_clip(frames: &[PathBuf], plates: &Path) -> Result<PathBuf, String> {
    let mut files: Vec<&PathBuf> = frames.iter().collect();
    files.dedup();
    let (stem, pattern) = clip_names(frames)?;
    let size = |p: &Path| std::fs::metadata(p).ok().filter(|m| m.is_file()).map(|m| m.len());
    let mut n = 1;
    let dir = loop {
        let d = if n == 1 { plates.join(&stem) } else { plates.join(format!("{stem}_{n}")) };
        let clash = files.iter().any(|f| {
            let there = f.file_name().map(|x| d.join(x));
            there.is_some_and(|t| size(&t).is_some_and(|s| Some(s) != size(f)))
        });
        if !clash {
            break d;
        }
        n += 1;
    };
    std::fs::create_dir_all(&dir).map_err(|e| format!("couldn't make the plates folder: {e}"))?;
    for f in files {
        let Some(name) = f.file_name() else { continue };
        let dst = dir.join(name);
        if size(&dst).is_some() {
            continue;
        }
        // Into a temp name, then renamed: a stopped copy leaves no half file.
        let tmp = dir.join(format!(".{}.copying", name.to_string_lossy()));
        std::fs::copy(f, &tmp).map_err(|e| format!("copying {}: {e}", f.display()))?;
        std::fs::rename(&tmp, &dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    }
    Ok(dir.join(pattern))
}

// ---- references ------------------------------------------------------------------

/// A reference to `rel` under `inst` in `p` — names only, no id: it is
/// found again by those names on whichever machine opens the comp.
pub fn reference(p: &Project, inst: Option<&Inst>, rel: &str) -> crate::assets::AssetRef {
    crate::assets::AssetRef { id: 0, project: p.name.clone(), shot: p.names(inst), rel: rel.to_string() }
}

fn find_in<'a>(ps: &'a [Project], r: &crate::assets::AssetRef) -> Option<PathBuf> {
    let p = ps.iter().find(|p| p.name == r.project).or_else(|| ps.iter().find(|p| !p.code.is_empty() && p.code == r.project))?;
    let inst = p.by_names(&r.shot)?;
    Some(join_rel(&p.path_of(inst), &r.rel))
}

/// Where a reference is on THIS machine: its project under the managed
/// root, its shot by name, its place under that.
pub fn resolve(r: &crate::assets::AssetRef) -> Result<String, String> {
    if r.rel.is_empty() {
        return Err(format!("asset {} has no place to find it by", r.id));
    }
    let present = |p: &Path| {
        if p.to_string_lossy().contains('#') {
            p.parent().is_some_and(Path::is_dir)
        } else {
            p.exists()
        }
    };
    let mut hit = find_in(&projects(false), r).filter(|p| present(p));
    if hit.is_none() {
        // A project added or a shot made since the last scan.
        hit = find_in(&projects(true), r).filter(|p| present(p));
    }
    match hit {
        Some(p) => Ok(p.to_string_lossy().into_owned()),
        None if managed_root().is_none() => {
            Err("no managed projects folder on this machine — set it in project… in the menubar".into())
        }
        None => Err(format!("{} is not in {} / {} on this machine", r.asset_label(), r.project_label(), r.shot_label())),
    }
}

/// Brings a clip into the shot whose plates folder is `plates`. With
/// organisicator running and knowing the project, IT copies and indexes
/// the clip (`POST /v1/ingest`), so the reference carries its id; otherwise
/// the files are copied here (`copy_clip`) and named by project and shot,
/// for organisicator to index on its next scan. Blocking: run it on a worker.
pub fn bring_in(src: &str, frames: &[PathBuf], plates: &Path) -> Result<crate::assets::AssetRef, String> {
    use crate::assets;
    let ps = projects(false);
    let (p, inst, under) = ps
        .iter()
        .find_map(|p| p.locate(plates).map(|(i, rel)| (p, i, rel)))
        .ok_or("that shot is not in a project")?;
    let api = assets::Client::discover()
        .ok()
        .and_then(|c| assets::api_ids(&c, &p.root, inst.map(|i| i.rel.as_str())).map(|ids| (c, ids)));
    match api {
        Some((client, (pid, iid))) => {
            let (stem, _) = clip_names(frames)?;
            let folder = if under.is_empty() { stem } else { format!("{under}/{stem}") };
            let dest = assets::Dest {
                project: pid,
                project_name: p.name.clone(),
                instance: iid,
                shot: p.names(inst),
                folder,
            };
            let r = assets::ingest(&client, src, &dest)?;
            assets::resolve(&client, &r).map(|res| res.named)
        }
        None => {
            let clip = copy_clip(frames, plates)?;
            locate_path(&clip).ok_or_else(|| "copied, but the copy is not inside a project".to_string())
        }
    }
}

/// A by-name reference with organisicator's id added when it is running
/// and has indexed the file, so it follows renames from then on. Unchanged
/// otherwise. Blocking (a local request or three).
pub fn with_id(r: crate::assets::AssetRef) -> crate::assets::AssetRef {
    use crate::assets;
    let Ok(client) = assets::Client::discover() else { return r };
    let ps = projects(false);
    let Some(p) = ps.iter().find(|p| p.name == r.project) else { return r };
    let Some(inst) = p.by_names(&r.shot) else { return r };
    let Some((pid, iid)) = assets::api_ids(&client, &p.root, inst.map(|i| i.rel.as_str())) else { return r };
    match assets::api_asset_id(&client, pid, iid, &r.rel) {
        Some(id) => assets::AssetRef { id, ..r },
        None => r,
    }
}

/// The project, level and place `path` is at, if it is in one.
pub fn locate_path(path: &Path) -> Option<crate::assets::AssetRef> {
    let ps = projects(false);
    ps.iter().find_map(|p| p.locate(path).map(|(i, rel)| reference(p, i, &rel)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = r#"{
        "format": 1, "app": "organisicator", "name": "spooky_beach", "code": "spb",
        "folders": [
            {"path": "sequences/{sequence}", "type": "sequence"},
            {"path": "sequences/{sequence}/{shot}", "type": "shot"},
            {"path": "sequences/{sequence}/{shot}/renders", "type": "render", "exts": "exr mov"},
            {"path": "sequences/{sequence}/{shot}/scripts", "type": "software", "exts": "nk blend"},
            {"path": "sequences/{sequence}/{shot}/plates", "type": "source", "exts": "exr"}
        ],
        "instances": [
            {"id": 1, "parent": null, "decl": "sequences/{sequence}", "value": "dev", "code": "dv", "rel": "sequences/dev"},
            {"id": 2, "parent": 1, "decl": "sequences/{sequence}/{shot}", "value": "spb_010", "code": "", "rel": "sequences/dev/spb_010"},
            {"id": 3, "parent": 1, "decl": "sequences/{sequence}/{shot}", "value": "spb_020", "code": "", "rel": "sequences/dev/spb_020"}
        ]
    }"#;

    fn fixture() -> (PathBuf, Project) {
        let root = std::env::temp_dir().join(format!("wobble-projects-{}-{:?}", std::process::id(), std::thread::current().id()));
        let _ = std::fs::remove_dir_all(&root);
        let proj = root.join("live").join("spooky_beach");
        std::fs::create_dir_all(proj.join(MANIFEST_DIR)).unwrap();
        std::fs::write(proj.join(MANIFEST_DIR).join(MANIFEST_FILE), MANIFEST).unwrap();
        let shot = proj.join("sequences/dev/spb_010");
        for sub in ["renders", "scripts", "plates/bg", "plates/fg"] {
            std::fs::create_dir_all(shot.join(sub)).unwrap();
        }
        std::fs::create_dir_all(proj.join("sequences/dev/spb_020/plates")).unwrap();
        let ps = find_projects(&root);
        assert_eq!(ps.len(), 1);
        (root, ps.into_iter().next().unwrap())
    }

    #[test]
    fn comps_go_in_the_software_folder_or_its_wobble_child() {
        let (root, p) = fixture();
        let shot = p.instance(2);
        let scripts = p.root.join("sequences").join("dev").join("spb_010").join("scripts");
        assert_eq!(p.comp_dir(shot), scripts);
        std::fs::create_dir_all(scripts.join("wobble")).unwrap();
        assert_eq!(p.comp_dir(shot), scripts.join("wobble"));
        // The sequence has no folders of its own: a comp folder in it.
        assert_eq!(p.comp_dir(p.instance(1)), p.root.join("sequences").join("dev").join("comp"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn new_comps_are_named_from_the_shot_and_never_overwrite() {
        let (root, p) = fixture();
        let shot = p.instance(2);
        let dir = p.comp_dir(shot);
        assert_eq!(new_comp_name(&p, shot, "comp", &dir), "spb_010_comp_v001.wbl");
        std::fs::write(dir.join("spb_010_comp_v001.wbl"), "{}").unwrap();
        assert_eq!(new_comp_name(&p, shot, "comp", &dir), "spb_010_comp_v002.wbl");
        assert_eq!(version_up(&dir.join("spb_010_comp_v001.wbl")).file_name().unwrap(), "spb_010_comp_v002.wbl");
        assert_eq!(next_version("a_v009"), "a_v010");
        assert_eq!(next_version("a"), "a_v002");
        assert_eq!(comps(&dir), vec![dir.join("spb_010_comp_v001.wbl")]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn media_groups_frames_and_leaves_other_shots_alone() {
        let (root, p) = fixture();
        let shot = p.path_of(p.instance(2));
        for f in 1..=3 {
            std::fs::write(shot.join(format!("plates/bg/bg.{f:04}.exr")), "").unwrap();
        }
        std::fs::write(shot.join("plates/fg/still_01.png"), "").unwrap();
        std::fs::write(shot.join("plates/fg/notes.txt"), "").unwrap();
        assert_eq!(media(&p, p.instance(2)), vec!["plates/bg/bg.####.exr", "plates/fg/still_01.png"]);
        // The sequence does not list its shots' plates as its own.
        assert!(media(&p, p.instance(1)).is_empty());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_clip_is_copied_into_the_plates_and_never_over_another() {
        let (root, p) = fixture();
        let src = root.join("outside");
        std::fs::create_dir_all(&src).unwrap();
        let frames: Vec<PathBuf> = (1..=3).map(|f| src.join(format!("bg.{f:04}.exr"))).collect();
        for f in &frames {
            std::fs::write(f, "abc").unwrap();
        }
        let shot = p.instance(2);
        let plates = p.plates_dir(shot);
        assert_eq!(plates, p.path_of(shot).join("plates"));
        let got = copy_clip(&frames, &plates).unwrap();
        assert_eq!(got, plates.join("bg").join("bg.####.exr"));
        assert_eq!(std::fs::read_to_string(plates.join("bg/bg.0002.exr")).unwrap(), "abc");
        // Again: the same files are kept, not copied twice.
        assert_eq!(copy_clip(&frames, &plates).unwrap(), got);
        // A different clip of the same name goes beside it.
        std::fs::write(&frames[0], "different").unwrap();
        assert_eq!(copy_clip(&frames, &plates).unwrap(), plates.join("bg_2").join("bg.####.exr"));
        assert_eq!(std::fs::read_to_string(plates.join("bg/bg.0001.exr")).unwrap(), "abc");
        // Found by name inside the shot.
        let (_, rel) = p.locate(&got).unwrap();
        assert_eq!(rel, "plates/bg/bg.####.exr");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_reference_is_names_and_finds_its_way_back() {
        let (root, p) = fixture();
        let shot = p.instance(2);
        let r = reference(&p, shot, "plates/bg/bg.####.exr");
        assert_eq!(r.shot, vec!["dev", "spb_010"]);
        assert_eq!(find_in(std::slice::from_ref(&p), &r), Some(join_rel(&p.path_of(shot), "plates/bg/bg.####.exr")));
        // By path, from anywhere inside the shot.
        let at = p.path_of(shot).join("plates").join("bg").join("bg.####.exr");
        let (i, rel) = p.locate(&at).unwrap();
        assert_eq!((i.map(|i| i.id), rel.as_str()), (Some(2), "plates/bg/bg.####.exr"));
        // A Windows-written place resolves with this OS's separators.
        let w = crate::assets::AssetRef { rel: "plates\\bg\\bg.####.exr".into(), ..r };
        assert_eq!(find_in(std::slice::from_ref(&p), &w), Some(at));
        let _ = std::fs::remove_dir_all(root);
    }
}
