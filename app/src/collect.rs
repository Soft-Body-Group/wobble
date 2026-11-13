//! Collect: copy every file a comp reads — Loads (sequences, movies,
//! stills), LUTs, font files, 3D scene files, .blend files — to one place,
//! then save the comp there with its paths pointing at the copies.
//!
//! Two destinations:
//! - a FOLDER: `<folder>/<name>.wbl` beside `<folder>/media/…`;
//! - an organisicator SHOT (or project): each Load's footage is INGESTED
//!   (`assets::ingest`, a copy organisicator indexes) and the Load then names
//!   the copy by asset id; the other files are copied into kind folders of
//!   the shot, and the comp is saved in its `comp` folder.
//!
//! Paths are found by walking the comp's JSON rather than matching node
//! kinds, so a font inside a 2dini body or a LUT inside a subwob is found
//! the same way as a Load at the top. Collect never touches the open comp
//! and never moves or overwrites an original.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::Value;

use crate::assets::{self, AssetRef, Dest};

/// What a path is for: picks its folder at the destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Footage,
    Lut,
    Font,
    Scene,
    Other,
}

impl Kind {
    pub fn folder(self) -> &'static str {
        match self {
            Kind::Footage => "footage",
            Kind::Lut => "luts",
            Kind::Font => "fonts",
            Kind::Scene => "3d",
            Kind::Other => "other",
        }
    }
}

/// How the authored string names its files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// `dir/name.####.exr`.
    Pattern,
    /// A directory of frames.
    Dir,
    /// One file (a movie, a still, a LUT, a font…).
    File,
}

/// One path the comp names, and the files behind it.
#[derive(Clone, Debug)]
pub struct Item {
    /// Exactly as the comp stores it (the key the rewrite matches on).
    pub path: String,
    pub kind: Kind,
    pub shape: Shape,
    pub files: Vec<PathBuf>,
    pub bytes: u64,
}

/// What a scan found: the items to copy and the paths left as they are.
#[derive(Clone, Debug, Default)]
pub struct Scan {
    pub items: Vec<Item>,
    /// (path, why) — remote, missing, unreadable.
    pub skipped: Vec<(String, String)>,
}

impl Scan {
    pub fn bytes(&self) -> u64 {
        self.items.iter().map(|i| i.bytes).sum()
    }
    pub fn files(&self) -> usize {
        self.items.iter().map(|i| i.files.len()).sum()
    }
}

// ---- finding paths ---------------------------------------------------------------

/// Every (path, kind) the comp's JSON names, deduped by path, in order.
/// A Render's `path` is an OUTPUT and is left out.
pub fn find_paths(graph: &Value) -> Vec<(String, Kind)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    walk(graph, Kind::Other, &mut |s, k| {
        if seen.insert(s.to_string()) {
            out.push((s.to_string(), k));
        }
    });
    out
}

/// Calls `f` on every string that names a file, with its kind. The rewrite
/// walks the same way (`walk_mut`), so what is found is what is repointed.
fn walk(v: &Value, ctx: Kind, f: &mut dyn FnMut(&str, Kind)) {
    match v {
        Value::Object(m) => {
            for (k, child) in m {
                match (k.as_str(), child) {
                    ("Write", _) => {}
                    ("path", Value::String(s)) | ("blend", Value::String(s)) if !s.is_empty() => {
                        f(s, if k == "blend" { Kind::Scene } else { ctx })
                    }
                    // A font file (`TextFont::File`).
                    ("File", Value::String(s)) if !s.is_empty() => f(s, Kind::Font),
                    _ => walk(child, child_kind(k, ctx), f),
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|c| walk(c, ctx, f)),
        _ => {}
    }
}

fn walk_mut(v: &mut Value, f: &mut dyn FnMut(&mut String)) {
    match v {
        Value::Object(m) => {
            for (k, child) in m.iter_mut() {
                match (k.as_str(), child) {
                    ("Write", _) => {}
                    ("path" | "blend" | "File", Value::String(s)) => f(s),
                    (_, child) => walk_mut(child, f),
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|c| walk_mut(c, f)),
        _ => {}
    }
}

fn child_kind(key: &str, ctx: Kind) -> Kind {
    match key {
        "Read" => Kind::Footage,
        "Lut" => Kind::Lut,
        // A Camera / Card's imported file (`scene3d::Source::File`).
        "File" => Kind::Scene,
        _ => ctx,
    }
}

/// The files behind every path the comp names. Reads the disk (a pattern
/// lists its directory), so run it on open, not per frame.
pub fn scan(graph: &Value) -> Scan {
    let mut s = Scan::default();
    for (path, kind) in find_paths(graph) {
        match files_of(&path, kind) {
            Ok((shape, files)) => {
                let bytes = files.iter().filter_map(|f| f.metadata().ok()).map(|m| m.len()).sum();
                s.items.push(Item { path, kind, shape, files, bytes });
            }
            Err(why) => s.skipped.push((path, why)),
        }
    }
    s
}

fn files_of(path: &str, kind: Kind) -> Result<(Shape, Vec<PathBuf>), String> {
    let p = engine::paths::expand_home(path);
    #[cfg(not(target_arch = "wasm32"))]
    if engine::objstore::is_remote(&p) {
        return Err("in object storage — left where it is".into());
    }
    if path.contains('#') {
        let (frames, _) = crate::resolve_pattern(&p).map_err(|e| format!("{e:#}"))?;
        if frames.is_empty() {
            return Err("no frames on disk".into());
        }
        return Ok((Shape::Pattern, frames));
    }
    if p.is_dir() {
        let mut frames: Vec<PathBuf> = std::fs::read_dir(&p)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|f| f.is_file())
            .filter(|f| kind != Kind::Footage || (engine::decode::is_supported(f) && !engine::decode::is_movie(f)))
            .collect();
        frames.sort();
        if frames.is_empty() {
            return Err("an empty folder".into());
        }
        return Ok((Shape::Dir, frames));
    }
    if p.is_file() {
        return Ok((Shape::File, vec![p]));
    }
    Err("not on disk".into())
}

// ---- planning copies -------------------------------------------------------------

/// Where one item lands: the copies to make and the string that names them.
#[derive(Clone, Debug)]
pub struct Placed {
    pub path: String,
    pub copies: Vec<(PathBuf, PathBuf)>,
}

/// Places `items` under `root`: a sequence in `root/<kind>/<its folder>/`,
/// a single file in `root/<kind>/`. Two sources never share a name.
pub fn place(items: &[&Item], root: &Path) -> Vec<Placed> {
    let mut used: HashSet<PathBuf> = HashSet::new();
    let mut unique = |want: PathBuf| -> PathBuf {
        if used.insert(want.clone()) {
            return want;
        }
        let stem = want.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let ext = want.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
        (2..)
            .map(|n| want.with_file_name(format!("{stem}_{n}{ext}")))
            .find(|c| used.insert(c.clone()))
            .unwrap()
    };
    items
        .iter()
        .map(|it| {
            let base = root.join(it.kind.folder());
            match it.shape {
                Shape::File => {
                    let src = &it.files[0];
                    let dst = unique(base.join(src.file_name().unwrap_or_default()));
                    Placed { path: dst.to_string_lossy().into_owned(), copies: vec![(src.clone(), dst)] }
                }
                Shape::Pattern | Shape::Dir => {
                    let p = engine::paths::expand_home(&it.path);
                    let folder = match it.shape {
                        Shape::Dir => p.file_name(),
                        _ => p.parent().and_then(|d| d.file_name()),
                    }
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "frames".into());
                    let dir = unique(base.join(folder));
                    let copies = it
                        .files
                        .iter()
                        .map(|f| (f.clone(), dir.join(f.file_name().unwrap_or_default())))
                        .collect();
                    let path = match it.shape {
                        Shape::Dir => dir.clone(),
                        _ => dir.join(p.file_name().unwrap_or_default()),
                    };
                    Placed { path: path.to_string_lossy().into_owned(), copies }
                }
            }
        })
        .collect()
}

// ---- the job ---------------------------------------------------------------------

/// Where the collect goes.
#[derive(Clone, Debug)]
pub enum Target {
    Folder(PathBuf),
    /// A shot picked in organisicator; footage goes in `dest.folder`.
    Project(Dest),
}

/// Progress a worker shares with the window.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
}

/// What came back: each old path's new one (and asset, for an ingested
/// Load), and where the comp goes.
#[derive(Debug, Default)]
pub struct Outcome {
    pub map: HashMap<String, (String, Option<AssetRef>)>,
    pub wbl: PathBuf,
    pub copied: usize,
    pub reused: usize,
}

/// Copies (or ingests) everything in `scan`. Blocking — run it on a worker.
pub fn run(scan: &Scan, target: &Target, name: &str, prog: &Progress) -> Result<Outcome, String> {
    prog.total.store(scan.bytes(), Ordering::Relaxed);
    let mut out = Outcome::default();
    let (root, client) = match target {
        Target::Folder(dir) => (dir.join("media"), None),
        Target::Project(dest) => {
            let client = assets::Client::discover()?;
            (shot_dir(&client, dest)?, Some((client, dest)))
        }
    };
    let wbl_dir = match target {
        Target::Folder(dir) => dir.clone(),
        Target::Project(_) => root.join("comp"),
    };
    out.wbl = wbl_dir.join(format!("{}.wbl", name.trim().trim_end_matches(".wbl")));

    // Footage into a project is organisicator's to copy: it indexes the
    // copy and answers its id.
    let mut rest: Vec<&Item> = Vec::new();
    for it in &scan.items {
        match &client {
            Some((client, dest)) if it.kind == Kind::Footage => {
                if prog.cancel.load(Ordering::Relaxed) {
                    return Err("stopped".into());
                }
                let src = engine::paths::expand_home(&it.path).to_string_lossy().into_owned();
                let (r, path) = ingest(client, &src, dest).map_err(|e| format!("{}: {e}", it.path))?;
                out.map.insert(it.path.clone(), (path, Some(r)));
                out.copied += it.files.len();
                prog.done.fetch_add(it.bytes, Ordering::Relaxed);
            }
            _ => rest.push(it),
        }
    }
    for (it, placed) in rest.iter().zip(place(&rest, &root)) {
        for (src, dst) in &placed.copies {
            if prog.cancel.load(Ordering::Relaxed) {
                return Err("stopped".into());
            }
            if copy_one(src, dst)? {
                out.copied += 1;
            } else {
                out.reused += 1;
            }
            prog.done.fetch_add(src.metadata().map(|m| m.len()).unwrap_or(0), Ordering::Relaxed);
        }
        out.map.insert(it.path.clone(), (placed.path, None));
    }
    std::fs::create_dir_all(&wbl_dir).map_err(|e| format!("couldn't make {}: {e}", wbl_dir.display()))?;
    Ok(out)
}

/// Copies one file; a copy already there at the same size is kept (a
/// second collect into the same folder only copies what changed). Returns
/// whether it copied.
fn copy_one(src: &Path, dst: &Path) -> Result<bool, String> {
    if src == dst {
        return Ok(false);
    }
    let len = src.metadata().map_err(|e| format!("{}: {e}", src.display()))?.len();
    if dst.metadata().is_ok_and(|m| m.is_file() && m.len() == len) {
        return Ok(false);
    }
    if let Some(d) = dst.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("couldn't make {}: {e}", d.display()))?;
    }
    // Into a temp name, then renamed: a stopped collect leaves no half file
    // that the size check above would keep.
    let tmp = dst.with_extension("collecting");
    std::fs::copy(src, &tmp).map_err(|e| format!("copying {}: {e}", src.display()))?;
    std::fs::rename(&tmp, dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    Ok(true)
}

/// The picked shot's folder on disk (the project's own for no shot).
fn shot_dir(client: &assets::Client, dest: &Dest) -> Result<PathBuf, String> {
    let projects = client.get("/v1/projects")?;
    let root = projects
        .as_array()
        .and_then(|a| a.iter().find(|p| p["id"].as_i64() == Some(dest.project)))
        .and_then(|p| p["root"].as_str())
        .ok_or_else(|| format!("organisicator has no project {}", dest.project_name))?;
    let mut dir = PathBuf::from(root);
    if let Some(inst) = dest.instance {
        let p = client.get(&format!("/v1/projects/{}", dest.project))?;
        let rel = p["instances"]
            .as_array()
            .and_then(|a| a.iter().find(|i| i["id"].as_i64() == Some(inst)))
            .and_then(|i| i["rel"].as_str())
            .ok_or("that shot is gone from organisicator")?;
        dir = dir.join(rel);
    }
    if !dir.is_dir() {
        return Err(format!("{} is offline", dir.display()));
    }
    Ok(dir)
}

/// `assets::ingest`, plus the path organisicator says the copy is at.
fn ingest(client: &assets::Client, src: &str, dest: &Dest) -> Result<(AssetRef, String), String> {
    let r = assets::ingest(client, src, dest)?;
    let path = assets::resolve(client, &r)?;
    Ok((path.named, path.path))
}

// ---- rewriting the comp ----------------------------------------------------------

/// Points every path in `graph` at its copy. A Load that was ingested names
/// its asset; any other Load that moved stops naming one (its copy is a
/// plain file, and an asset would send the next load back to the original).
pub fn rewrite(graph: &mut Value, map: &HashMap<String, (String, Option<AssetRef>)>) {
    fix_assets(graph, map);
    walk_mut(graph, &mut |s| {
        if let Some((new, _)) = map.get(s.as_str()) {
            *s = new.clone();
        }
    });
}

fn fix_assets(v: &mut Value, map: &HashMap<String, (String, Option<AssetRef>)>) {
    match v {
        Value::Object(m) => {
            let read = m.get("kind").and_then(|k| k.get("Read")).and_then(|r| r.get("path")).and_then(|p| p.as_str());
            if let Some((_, asset)) = read.and_then(|p| map.get(p)) {
                match asset {
                    Some(a) => {
                        m.insert("asset".into(), serde_json::to_value(a).unwrap_or(Value::Null));
                    }
                    None => {
                        m.remove("asset");
                    }
                }
            }
            for (k, child) in m.iter_mut() {
                if k != "Write" {
                    fix_assets(child, map);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|c| fix_assets(c, map)),
        _ => {}
    }
}

/// Human size: `1.2 GB`.
pub fn size(bytes: u64) -> String {
    let b = bytes as f64;
    match bytes {
        0..1_000_000 => format!("{:.0} KB", b / 1e3),
        1_000_000..1_000_000_000 => format!("{:.1} MB", b / 1e6),
        _ => format!("{:.2} GB", b / 1e9),
    }
}

// ---- the window ------------------------------------------------------------------

/// The collect window's state, while it is up.
pub struct Window {
    pub scan: Scan,
    pub to_project: bool,
    pub folder: String,
    pub name: String,
    pub dest: Option<Dest>,
    pub picker: Option<assets::Picker>,
    pub job: Option<(Arc<Progress>, std::sync::mpsc::Receiver<Result<Outcome, String>>)>,
    pub msg: Option<(bool, String)>,
    /// The comp it saved, for "open it".
    pub saved: Option<PathBuf>,
}

/// What the window asks the app to do this frame.
pub enum Ask {
    None,
    Start(Target),
    Open(PathBuf),
}

impl Window {
    pub fn new(graph: &Value, current: Option<&Path>) -> Self {
        let name = current
            .and_then(|p| p.file_stem())
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "comp".into());
        let folder = current
            .and_then(|p| p.parent())
            .map(|d| d.join(format!("{name}_collected")).to_string_lossy().into_owned())
            .unwrap_or_default();
        Window {
            scan: scan(graph),
            to_project: false,
            folder,
            name,
            dest: None,
            picker: None,
            job: None,
            msg: None,
            saved: None,
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context, open: &mut bool) -> Ask {
        let mut ask = Ask::None;
        let busy = self.job.is_some();
        egui::Window::new("collect")
            .open(open)
            .default_width(420.0)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.weak("copies every file this comp reads to one place and saves the comp there, pointing at the copies. The open comp is not changed.");
                ui.add_space(4.0);
                let s = &self.scan;
                ui.label(format!("{} files in {} paths, {}", s.files(), s.items.len(), size(s.bytes())));
                egui::CollapsingHeader::new("what it reads").id_salt("collect-list").show(ui, |ui| {
                    egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                        for it in &s.items {
                            ui.label(format!("{} · {} · {}", it.kind.folder(), it.path, size(it.bytes)))
                                .on_hover_text(format!("{} files", it.files.len()));
                        }
                        for (p, why) in &s.skipped {
                            ui.colored_label(crate::theme::warn(), format!("left as is: {p} ({why})"));
                        }
                    });
                });
                ui.separator();
                ui.add_enabled_ui(!busy, |ui| {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.to_project, false, "to a folder");
                        ui.selectable_value(&mut self.to_project, true, "to an organisicator shot");
                    });
                    ui.horizontal(|ui| {
                        ui.label("name");
                        ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(160.0));
                        ui.weak(".wbl");
                    });
                    if self.to_project {
                        let label = match &self.dest {
                            Some(d) => format!(
                                "{} / {} (footage in {})",
                                d.project_name,
                                if d.shot.is_empty() { "(project)".into() } else { d.shot.join(" › ") },
                                d.folder
                            ),
                            None => "no shot picked".into(),
                        };
                        ui.horizontal(|ui| {
                            if ui.button("pick shot…").clicked() {
                                self.picker = Some(assets::Picker::for_collect());
                            }
                            ui.label(label);
                        });
                        ui.weak("footage is ingested (organisicator copies and indexes it, Loads name it by id); LUTs, fonts and 3D files go in folders of the shot; the comp goes in its comp folder");
                    } else {
                        ui.horizontal(|ui| {
                            ui.label("folder");
                            ui.add(egui::TextEdit::singleline(&mut self.folder).desired_width(250.0));
                            if ui.button("…").clicked() {
                                if let Some(d) = crate::filedialog::FileDialog::new().pick_folder() {
                                    self.folder = d.to_string_lossy().into_owned();
                                }
                            }
                        });
                    }
                });
                ui.add_space(4.0);
                if let Some((prog, _)) = &self.job {
                    let (d, t) = (prog.done.load(Ordering::Relaxed), prog.total.load(Ordering::Relaxed).max(1));
                    ui.add(egui::ProgressBar::new(d as f32 / t as f32).text(format!("{} of {}", size(d), size(t))));
                    if ui.button("stop").clicked() {
                        prog.cancel.store(true, Ordering::Relaxed);
                    }
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                } else {
                    let target = if self.to_project {
                        self.dest.clone().map(Target::Project)
                    } else {
                        Some(self.folder.trim()).filter(|f| !f.is_empty()).map(|f| Target::Folder(engine::paths::expand_home(f)))
                    };
                    let ok = target.is_some() && !self.name.trim().is_empty();
                    ui.horizontal(|ui| {
                        if ui.add_enabled(ok, egui::Button::new("collect")).clicked() {
                            if let Some(t) = target {
                                ask = Ask::Start(t);
                            }
                        }
                        if let Some(p) = &self.saved {
                            if ui.button("open it").on_hover_text(p.display().to_string()).clicked() {
                                ask = Ask::Open(p.clone());
                            }
                        }
                    });
                }
                if let Some((good, m)) = &self.msg {
                    let c = if *good { ui.visuals().text_color() } else { crate::theme::error() };
                    ui.colored_label(c, m);
                }
            });
        if let Some(mut p) = self.picker.take() {
            let mut open = true;
            match p.ui(ctx, &mut open) {
                Some(assets::Picked::IngestTo(d)) => {
                    self.dest = Some(d);
                    open = false;
                }
                Some(assets::Picked::Asset(_)) | None => {}
            }
            if open {
                self.picker = Some(p);
            }
        }
        ask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn comp(plate: &str, lut: &str, font: &str, out: &str) -> Value {
        json!({"nodes": {
            "1": {"kind": {"Read": {"source": 0, "path": plate}}, "asset": {"id": 4}},
            "2": {"kind": {"Lut": {"path": lut}}},
            "3": {"kind": {"Geo": {"graph": {"nodes": {"1": {"Text": {"font": {"File": font}}}}}}}},
            "4": {"kind": {"Write": {"path": out, "location": out}}},
            "5": {"kind": {"Camera": {"source": {"File": {"path": "/no/such.usda", "object": ""}}}}},
        }})
    }

    #[test]
    fn every_kind_of_path_is_found_and_a_render_is_not() {
        let g = comp("/p/bg.####.exr", "/l/a.cube", "/f/x.ttf", "/out");
        let found = find_paths(&g);
        assert_eq!(
            found,
            vec![
                ("/p/bg.####.exr".to_string(), Kind::Footage),
                ("/l/a.cube".to_string(), Kind::Lut),
                ("/f/x.ttf".to_string(), Kind::Font),
                ("/no/such.usda".to_string(), Kind::Scene),
            ]
        );
    }

    #[test]
    fn a_collect_copies_the_frames_and_the_comp_points_at_them() {
        let tmp = std::env::temp_dir().join(format!("wobble-collect-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let plates = tmp.join("src/bg_v001");
        std::fs::create_dir_all(&plates).unwrap();
        for f in 1001..=1003 {
            std::fs::write(plates.join(format!("bg.{f}.exr")), vec![f as u8; 10]).unwrap();
        }
        let lut = tmp.join("src/look.cube");
        std::fs::write(&lut, b"LUT_3D_SIZE 2").unwrap();
        let pattern = plates.join("bg.####.exr").to_string_lossy().into_owned();
        let mut g = comp(&pattern, &lut.to_string_lossy(), "", "/out");
        let s = scan(&g);
        assert_eq!((s.items.len(), s.files()), (2, 4), "{:?}", s.skipped);
        assert_eq!(s.skipped.len(), 1, "the missing scene file is reported, not fatal");

        let dst = tmp.join("dst");
        let prog = Progress::default();
        let out = run(&s, &Target::Folder(dst.clone()), "shot", &prog).unwrap();
        assert_eq!((out.copied, out.reused), (4, 0));
        assert_eq!(out.wbl, dst.join("shot.wbl"));
        assert_eq!(prog.done.load(Ordering::Relaxed), s.bytes());
        assert!(dst.join("media/footage/bg_v001/bg.1002.exr").is_file());
        assert!(dst.join("media/luts/look.cube").is_file());
        assert!(plates.join("bg.1002.exr").is_file(), "a copy, never a move");

        rewrite(&mut g, &out.map);
        // Built with `join`, never a "/" literal: on Windows the collect's
        // own paths come out with backslashes.
        let new_plate = dst
            .join("media")
            .join("footage")
            .join("bg_v001")
            .join("bg.####.exr")
            .to_string_lossy()
            .into_owned();
        assert_eq!(g["nodes"]["1"]["kind"]["Read"]["path"], json!(new_plate));
        assert!(g["nodes"]["1"].get("asset").is_none(), "a copied Load no longer names the original asset");
        assert_eq!(g["nodes"]["4"]["kind"]["Write"]["path"], json!("/out"), "a render is not repointed");

        // Again into the same folder: nothing new to copy.
        let again = run(&s, &Target::Folder(dst.clone()), "shot", &Progress::default()).unwrap();
        assert_eq!((again.copied, again.reused), (0, 4));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn two_sources_with_one_name_land_apart() {
        let a = Item { path: "/a/x.cube".into(), kind: Kind::Lut, shape: Shape::File, files: vec!["/a/x.cube".into()], bytes: 1 };
        let b = Item { path: "/b/x.cube".into(), kind: Kind::Lut, shape: Shape::File, files: vec!["/b/x.cube".into()], bytes: 1 };
        let p = place(&[&a, &b], Path::new("/r"));
        let at = |name: &str| Path::new("/r").join("luts").join(name).to_string_lossy().into_owned();
        assert_eq!(p[0].path, at("x.cube"));
        assert_eq!(p[1].path, at("x_2.cube"));
    }

    /// The JSON shapes the walk assumes are the ones the real graph writes.
    #[test]
    fn the_real_graph_round_trips_repointed() {
        use engine::graph::NodeKind;
        let mut g = crate::editor::EditorGraph::default();
        let id = g.add(
            NodeKind::Read { source: 0, path: "/p/bg.####.exr".into(), range: None, ram_cache: false, nvme_cache: true, layer: None, input: Default::default(), multi: false, start: None, first: None, last: None, edges: Default::default(), cache_on_play: true, single: Default::default(), eight_bit: false },
            egui::Pos2::ZERO,
        );
        g.nodes.get_mut(&id).unwrap().asset = Some(AssetRef { id: 4, ..Default::default() });
        let mut v = serde_json::to_value(&g).unwrap();
        assert_eq!(find_paths(&v), vec![("/p/bg.####.exr".to_string(), Kind::Footage)]);
        let map = HashMap::from([("/p/bg.####.exr".to_string(), ("/c/bg.####.exr".to_string(), None))]);
        rewrite(&mut v, &map);
        let back: crate::editor::EditorGraph = serde_json::from_value(v).unwrap();
        let n = &back.nodes[&id];
        assert!(matches!(&n.kind, NodeKind::Read { path, .. } if path == "/c/bg.####.exr"));
        assert_eq!(n.asset, None);
    }

    #[test]
    fn an_ingested_load_names_its_asset() {
        let mut g = comp("/p/bg.####.exr", "", "", "");
        let r = AssetRef { id: 77, project: "p".into(), shot: vec!["a".into()], rel: "plates/bg.####.exr".into() };
        let map = HashMap::from([("/p/bg.####.exr".to_string(), ("/m/a/plates/bg.####.exr".to_string(), Some(r.clone())))]);
        rewrite(&mut g, &map);
        assert_eq!(g["nodes"]["1"]["asset"]["id"], json!(77));
        assert_eq!(g["nodes"]["1"]["kind"]["Read"]["path"], json!("/m/a/plates/bg.####.exr"));
    }
}
