//! Organisicator marketing posts. A marketing folder holds `rushes/` (raw
//! captures), `posts/<slug>/post.json` (one per post) and `result/`
//! (finished media, `<slug>_v001.mp4`…), as written by organisicator's
//! `src-tauri/src/marketing.rs`. An ADVANCED post is edited in other
//! software and gets `scripts/`, `in/` and `out/`; its renders are published
//! from `out/`.
//!
//! Scrub cuts advanced posts: opening one opens (or starts) the post's Scrub
//! project, `posts/<slug>/<slug>.scrub` BESIDE `post.json`, so the post
//! folder holds the edit as a simple post's `post.json` holds its cut, and
//! organisicator's own folders (`scripts`, `in`, `out`) keep their meaning.
//! The sequence takes the post's frame shape (16:9, 9:16, 1:1, 4:5), the bin
//! its sources and `in/` files, the browser its `rushes/`. Renders go to
//! `out/` and, when asked, `result/`.
//!
//! `post.json` is organisicator's. Scrub reads it, and writes exactly ONE
//! field, `frame` (the post's frame shape), when the user sets it from
//! sequence settings: read, change that key, write the rest back as it was
//! (unknown fields and key order kept), a hidden backup beside it, temp
//! file + rename. Clip settings, results and the publish record are never
//! touched; copying renders into `result/` is organisicator's publish step.

use crate::editor::Editor;
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const RUSHES: &str = "rushes";
pub const POSTS: &str = "posts";
pub const RESULT: &str = "result";
pub const POST_FILE: &str = "post.json";

/// The fields of `post.json` Scrub reads.
#[derive(Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PostDoc {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub frame: String,
    /// Rushes an advanced post uses (relative to the marketing folder).
    #[serde(default)]
    pub sources: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Post {
    pub slug: String,
    pub doc: PostDoc,
    pub problem: Option<String>,
}

impl Post {
    pub fn advanced(&self) -> bool {
        self.doc.kind == "advanced"
    }
}

/// An open post: where it lives and what it is.
#[derive(Debug, Clone, PartialEq)]
pub struct PostCtx {
    /// The marketing folder.
    pub mdir: PathBuf,
    pub slug: String,
    pub title: String,
    pub frame: String,
}

/// The raster a post's frame shape asks for; `None` for "original" (the
/// first clip decides, as for any sequence).
pub fn raster_for(frame: &str) -> Option<(u32, u32)> {
    match frame {
        "16:9" => Some((1920, 1080)),
        "9:16" => Some((1080, 1920)),
        "1:1" => Some((1080, 1080)),
        "4:5" => Some((1080, 1350)),
        _ => None,
    }
}

/// The frame shapes a post can have, and their rasters.
pub const SHAPES: [(&str, (u32, u32)); 4] = [("16:9", (1920, 1080)), ("9:16", (1080, 1920)), ("1:1", (1080, 1080)), ("4:5", (1080, 1350))];

/// The frame shape a raster is (within 1%), if any.
pub fn shape_of(raster: (u32, u32)) -> Option<&'static str> {
    let a = raster.0 as f64 / raster.1.max(1) as f64;
    SHAPES.iter().find(|(_, (w, h))| (*w as f64 / *h as f64 - a).abs() / a < 0.01).map(|(s, _)| *s)
}

/// Set a post's `frame` and nothing else (see the module doc).
pub fn write_frame(post_dir: &Path, frame: &str) -> Result<(), String> {
    let path = post_dir.join(POST_FILE);
    let text = std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let mut v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{} is not JSON: {e}", path.display()))?;
    let obj = v.as_object_mut().ok_or_else(|| format!("{} is not an object", path.display()))?;
    if obj.get("frame").and_then(|f| f.as_str()) == Some(frame) {
        return Ok(());
    }
    obj.insert("frame".into(), serde_json::Value::String(frame.into()));
    let mut out = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
    if text.ends_with('\n') {
        out.push('\n');
    }
    crate::project::write_atomic(&post_dir.join(format!(".{POST_FILE}.scrub-backup")), text.as_bytes()).map_err(|e| format!("backing up {}: {e}", path.display()))?;
    crate::project::write_atomic(&path, out.as_bytes()).map_err(|e| format!("writing {}: {e}", path.display()))
}

/// Is `dir` a marketing folder (has `posts/` and `rushes/` or `result/`)?
pub fn is_marketing(dir: &Path) -> bool {
    dir.join(POSTS).is_dir() && (dir.join(RUSHES).is_dir() || dir.join(RESULT).is_dir())
}

/// The marketing folder holding `path`, walking up.
pub fn find(path: &Path) -> Option<PathBuf> {
    let mut d = if path.is_dir() { Some(path) } else { path.parent() };
    while let Some(dir) = d {
        if is_marketing(dir) {
            return Some(dir.to_path_buf());
        }
        d = dir.parent();
    }
    None
}

/// The marketing folder organisicator's recorders write to, if set
/// (`record-target.json`, `marketing` field).
pub fn record_target() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let p = home.join("Library/Application Support/com.alex.organisicator/record-target.json");
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(p).ok()?).ok()?;
    let m = PathBuf::from(v.get("marketing")?.as_str()?);
    is_marketing(&m).then_some(m)
}

/// Every post in a marketing folder, sorted by name (a broken `post.json`
/// is listed with its problem).
pub fn list_posts(mdir: &Path) -> Vec<Post> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(mdir.join(POSTS)) else { return out };
    for e in rd.flatten() {
        if !e.path().is_dir() {
            continue;
        }
        let slug = e.file_name().to_string_lossy().into_owned();
        if slug.starts_with('.') {
            continue;
        }
        let (doc, problem) = match std::fs::read(e.path().join(POST_FILE)).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<PostDoc>(&b).map_err(|e| e.to_string())) {
            Ok(d) => (d, None),
            Err(e) => (PostDoc { title: slug.clone(), ..Default::default() }, Some(e)),
        };
        out.push(Post { slug, doc, problem });
    }
    out.sort_by(|a, b| a.slug.cmp(&b.slug));
    out
}

/// `name` is version N of `slug` (`teaser_v003.mp4` -> 3), as organisicator reads results.
pub fn result_version(slug: &str, name: &str) -> Option<u32> {
    let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
    let digits = stem.strip_prefix(slug)?.strip_prefix("_v")?;
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then(|| digits.parse().ok()).flatten()
}

impl PostCtx {
    pub fn dir(&self) -> PathBuf {
        self.mdir.join(POSTS).join(&self.slug)
    }

    /// The post's Scrub project: beside its `post.json` (`<slug>.scrub`, or
    /// the `<slug>.edify` an older build made there).
    pub fn project_path(&self) -> PathBuf {
        let new = self.dir().join(format!("{}.{}", self.slug, crate::project::EXT));
        let old = self.dir().join(format!("{}.{}", self.slug, crate::project::OLD_EXT));
        if !new.exists() && old.is_file() { old } else { new }
    }

    /// The next free render in `out/`: `<slug>_<preset>_v001.mp4`…
    pub fn out_path(&self, preset: &str) -> PathBuf {
        let out = self.dir().join("out");
        (1..10_000)
            .map(|v| out.join(format!("{}_{preset}_v{v:03}.mp4", self.slug)))
            .find(|p| !p.exists())
            .unwrap_or_else(|| out.join(format!("{}_{preset}.mp4", self.slug)))
    }

    /// The post's next result: `result/<slug>_v<NNN>.<ext>`, one past the
    /// highest version there (organisicator's `next_result_path`). Scrub
    /// does not write results (organisicator's publish step does); this is
    /// for showing what the next one will be called.
    #[allow(dead_code)]
    pub fn next_result(&self, ext: &str) -> PathBuf {
        let top = std::fs::read_dir(self.mdir.join(RESULT))
            .map(|rd| rd.flatten().filter_map(|e| result_version(&self.slug, &e.file_name().to_string_lossy())).max().unwrap_or(0))
            .unwrap_or(0);
        self.mdir.join(RESULT).join(format!("{}_v{:03}.{ext}", self.slug, top + 1))
    }

    /// The post a project file at `path` belongs to (`posts/<slug>/<slug>.scrub`, or an older `.edify`).
    pub fn for_project(path: &Path) -> Option<PostCtx> {
        let dir = path.parent()?;
        let slug = dir.file_name()?.to_string_lossy().into_owned();
        let posts = dir.parent()?;
        if posts.file_name()? != POSTS {
            return None;
        }
        let mdir = posts.parent()?.to_path_buf();
        if !is_marketing(&mdir) {
            return None;
        }
        let doc: PostDoc = std::fs::read(dir.join(POST_FILE)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
        Some(PostCtx { mdir, title: if doc.title.is_empty() { slug.clone() } else { doc.title }, frame: doc.frame, slug })
    }
}

/// The "open marketing post" window's state.
pub struct PostsWindow {
    pub dir: String,
    pub posts: Vec<Post>,
    listed: Option<PathBuf>,
}

impl Editor {
    /// Set the open post's frame shape: written to its `post.json` (that
    /// field only) and the sequence raster follows.
    pub fn set_post_frame(&mut self, frame: &str) -> Result<(), String> {
        let Some(p) = &self.post else { return Err("not in a marketing post".into()) };
        write_frame(&p.dir(), frame)?;
        if let Some(r) = raster_for(frame) {
            self.rec.seq.raster = Some(r);
            self.changed(false);
        }
        if let Some(p) = &mut self.post {
            p.frame = frame.to_string();
        }
        self.say(format!("the post's frame shape is now {frame}"));
        Ok(())
    }

    pub fn open_posts_window(&mut self) {
        let guess = self
            .post
            .as_ref()
            .map(|p| p.mdir.clone())
            .or_else(|| self.project.path.as_deref().and_then(find))
            .or_else(|| self.first_media().as_deref().and_then(find))
            .or_else(record_target)
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        self.posts_window = Some(PostsWindow { dir: guess, posts: Vec::new(), listed: None });
    }

    /// Open an advanced post: its project if it has one, else a new project
    /// shaped by the post (saved beside `post.json` on the first save). The
    /// bins: the post's own rushes and `in/` files first ("post: …"), then
    /// every folder of `rushes` (the project's rushes folders; the post's
    /// marketing folder's when empty), listed in the background.
    pub fn open_post(&mut self, mdir: &Path, post: &Post, rushes: &[PathBuf]) {
        let ctx = PostCtx { mdir: mdir.to_path_buf(), slug: post.slug.clone(), title: post.doc.title.clone(), frame: post.doc.frame.clone() };
        let project = ctx.project_path();
        let fresh = !project.exists();
        if fresh {
            let mut seq = model::Sequence::new(self.rec.seq.fps);
            seq.raster = raster_for(&post.doc.frame);
            self.new_project(seq, Some(project));
        } else {
            self.open_project(&project);
        }
        let title = if post.doc.title.is_empty() { post.slug.clone() } else { post.doc.title.clone() };
        let own = format!("{}{title}", crate::bin::POST_PREFIX);
        let mut add: Vec<PathBuf> = post.doc.sources.iter().map(|s| mdir.join(s)).collect();
        if let Ok(rd) = std::fs::read_dir(ctx.dir().join("in")) {
            let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_file() && !p.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.'))).collect();
            v.sort();
            add.extend(v);
        }
        for p in add.iter().filter(|p| p.exists()) {
            self.bin.add_to(p, String::new(), &own);
        }
        let dirs: Vec<PathBuf> = if rushes.is_empty() { vec![mdir.join(RUSHES)] } else { rushes.to_vec() };
        let several = dirs.len() > 1;
        let labelled = dirs
            .into_iter()
            .filter(|d| d.is_dir())
            .map(|d| {
                let label = if several { d.parent().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default() } else { String::new() };
                (d, label)
            })
            .collect();
        self.bin.scan_rushes(labelled);
        self.bin.go(mdir.join(RUSHES));
        self.focus = crate::editor::Focus::Bin;
        if fresh {
            self.say(format!("new edit for post \"{title}\" ({}): cmd S saves it beside post.json", if post.doc.frame.is_empty() { "original" } else { &post.doc.frame }));
        }
        self.post = Some(ctx);
    }

    /// `--post DIR` (a post folder or its post.json), as organisicator
    /// launches Scrub: open that post with `rushes` in the bins.
    pub fn open_post_path(&mut self, path: &Path, rushes: &[PathBuf]) -> Result<(), String> {
        let dir = if path.is_file() { path.parent().unwrap_or(path) } else { path };
        let slug = dir.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or("not a post folder")?;
        let mdir = dir.parent().filter(|p| p.file_name().is_some_and(|n| n == POSTS)).and_then(|p| p.parent()).ok_or_else(|| format!("{} is not inside a marketing folder's posts/", dir.display()))?;
        if !is_marketing(mdir) {
            return Err(format!("{} is not a marketing folder", mdir.display()));
        }
        let post = list_posts(mdir).into_iter().find(|p| p.slug == slug).ok_or_else(|| format!("no post {slug} in {}", mdir.display()))?;
        if let Some(e) = &post.problem {
            return Err(format!("{}: {e}", dir.join(POST_FILE).display()));
        }
        self.open_post(&mdir.to_path_buf(), &post, rushes);
        Ok(())
    }

    pub fn posts_ui(&mut self, ctx: &egui::Context) {
        let Some(w) = &mut self.posts_window else { return };
        let dir = imfram::filebrowser::expand_home(&w.dir);
        if w.listed.as_ref() != Some(&dir) {
            w.posts = if is_marketing(&dir) { list_posts(&dir) } else { Vec::new() };
            w.listed = Some(dir.clone());
        }
        let mut open = true;
        let mut pick: Option<Post> = None;
        egui::Window::new("open marketing post").open(&mut open).default_width(520.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("marketing folder");
                ui.text_edit_singleline(&mut w.dir);
                if ui.button("choose…").clicked() {
                    if let Some(d) = imfram::filedialog::FileDialog::new().pick_folder() {
                        w.dir = d.display().to_string();
                    }
                }
            });
            if !is_marketing(&dir) {
                ui.weak("not a marketing folder (it needs posts/ and rushes/ or result/)");
                return;
            }
            if w.posts.is_empty() {
                ui.weak("no posts yet: make one in organisicator");
            }
            for p in &w.posts {
                ui.horizontal(|ui| {
                    let frame = if p.doc.frame.is_empty() { "original" } else { &p.doc.frame };
                    let label = format!("{}   {}   {}", p.doc.title, p.doc.kind, frame);
                    if p.advanced() {
                        if ui.button("open").clicked() {
                            pick = Some(p.clone());
                        }
                        ui.label(label);
                    } else {
                        ui.add_enabled(false, egui::Button::new("open"));
                        ui.weak(format!("{label}  (simple posts are cut in organisicator)"));
                    }
                    if let Some(e) = &p.problem {
                        ui.colored_label(imfram::theme::error(), e);
                    }
                });
            }
        });
        if let Some(p) = pick {
            self.posts_window = None;
            self.open_post(&dir, &p, &[]);
            return;
        }
        if !open {
            self.posts_window = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(tag: &str) -> PathBuf {
        let m = std::env::temp_dir().join(format!("scrub-mkt-{tag}-{}", std::process::id())).join("marketing");
        let _ = std::fs::remove_dir_all(&m);
        for d in ["rushes/2026-11-11", "posts/teaser/in", "posts/teaser/out", "posts/teaser/scripts", "posts/quick", "result"] {
            std::fs::create_dir_all(m.join(d)).unwrap();
        }
        std::fs::write(
            m.join("posts/teaser/post.json"),
            r#"{"title": "Teaser", "kind": "advanced", "created": 1, "frame": "9:16", "sources": ["rushes/2026-11-11/a.mp4"], "unknown": 3}"#,
        )
        .unwrap();
        std::fs::write(m.join("posts/quick/post.json"), r#"{"title": "Quick", "kind": "simple", "frame": "original"}"#).unwrap();
        std::fs::write(m.join("result/teaser_v001.mp4"), b"x").unwrap();
        std::fs::write(m.join("result/teaser_v007.mp4"), b"x").unwrap();
        std::fs::write(m.join("result/other_v009.mp4"), b"x").unwrap();
        m
    }

    #[test]
    fn posts_are_listed_read_only_with_their_shape() {
        let m = fixture("list");
        let before = std::fs::read(m.join("posts/teaser/post.json")).unwrap();
        let posts = list_posts(&m);
        assert_eq!(posts.iter().map(|p| p.slug.as_str()).collect::<Vec<_>>(), ["quick", "teaser"]);
        let t = &posts[1];
        assert!(t.advanced());
        assert_eq!(raster_for(&t.doc.frame), Some((1080, 1920)));
        assert_eq!(t.doc.sources, ["rushes/2026-11-11/a.mp4"]);
        assert_eq!(std::fs::read(m.join("posts/teaser/post.json")).unwrap(), before, "never written");
        let _ = std::fs::remove_dir_all(m.parent().unwrap());
    }

    #[test]
    fn writing_the_frame_changes_that_field_only() {
        let m = fixture("frame");
        let dir = m.join("posts/teaser");
        write_frame(&dir, "4:5").unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(POST_FILE)).unwrap()).unwrap();
        assert_eq!(v["frame"], "4:5");
        assert_eq!(v["unknown"], 3, "unknown fields kept");
        assert_eq!(v["sources"][0], "rushes/2026-11-11/a.mp4");
        assert_eq!(v["created"], 1);
        let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        assert_eq!(keys, ["title", "kind", "created", "frame", "sources", "unknown"], "key order kept");
        let backup = std::fs::read_to_string(dir.join(".post.json.scrub-backup")).unwrap();
        assert!(backup.contains("\"9:16\""), "the backup is the file as it was");
        assert_eq!(shape_of((1080, 1350)), Some("4:5"));
        assert_eq!(shape_of((1000, 700)), None);
        let _ = std::fs::remove_dir_all(m.parent().unwrap());
    }

    #[test]
    fn renders_go_to_out_and_the_next_result_version() {
        let m = fixture("paths");
        let p = PostCtx { mdir: m.clone(), slug: "teaser".into(), title: "Teaser".into(), frame: "9:16".into() };
        assert_eq!(p.project_path(), m.join("posts/teaser/teaser.scrub"));
        assert_eq!(p.out_path("reels"), m.join("posts/teaser/out/teaser_reels_v001.mp4"));
        assert_eq!(p.next_result("mp4"), m.join("result/teaser_v008.mp4"), "one past the highest version");
        assert_eq!(PostCtx::for_project(&p.project_path()).map(|c| c.slug), Some("teaser".into()));
        assert_eq!(find(&m.join("posts/teaser/in")), Some(m.clone()));
        let _ = std::fs::remove_dir_all(m.parent().unwrap());
    }
}
