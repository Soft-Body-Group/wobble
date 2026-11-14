//! Organisicator, through the one surface it publishes on every machine:
//! `<project>/.organisicator/project.json`, the project's structure with
//! every path relative to its root (written by organisicator's
//! `src-tauri/src/manifest.rs`, read the same way by its Blender addon,
//! `blender_addon/organisicator/core.py`, whose rules this mirrors).
//!
//! What Scrub does with it:
//! - **Knows where it is**: from any file (a clip, a project) it finds the
//!   enclosing organisicator project and the deepest shot/sequence
//!   INSTANCE holding it.
//! - **Saves where the project expects**: an untitled project's save
//!   location is the instance's Scrub folder (a folder named Scrub, else one
//!   whose file types include `.scrub` (or `.edify`), else its software folder or that
//!   folder's `Scrub` child), with a versioned name `<shot>_edit_v001.scrub`.
//! - **Jumps the browser** to the instance's folders (plates, renders…).
//!
//! The manifest is DERIVED data that organisicator rewrites; Scrub only
//! reads it and never writes into `.organisicator/`. Organisicator's local
//! HTTP API is not used: it exists on a branch, not on main (see
//! docs/organisicator.md).

use serde::Deserialize;
use std::path::{Path, PathBuf};

pub const DIR: &str = ".organisicator";
pub const FILE: &str = "project.json";
/// The manifest formats this reader understands.
pub const FORMAT: u32 = 1;

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Folder {
    /// Anatomy path, `{variable}` segments for repeating levels.
    pub path: String,
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub exts: String,
    #[serde(default)]
    pub hint: String,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Instance {
    pub id: i64,
    pub parent: Option<i64>,
    pub decl: String,
    pub value: String,
    #[serde(default)]
    pub code: String,
    /// Concrete path under the project root, `/`-separated.
    pub rel: String,
}

#[derive(Deserialize, Debug, Clone, PartialEq)]
pub struct Manifest {
    pub format: u32,
    pub name: String,
    #[serde(default)]
    pub code: String,
    #[serde(default)]
    pub folders: Vec<Folder>,
    #[serde(default)]
    pub instances: Vec<Instance>,
}

/// A project found on disk: its root and its manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct Project {
    pub root: PathBuf,
    pub manifest: Manifest,
}

fn segs(rel: &str) -> Vec<&str> {
    rel.split(['/', '\\']).filter(|s| !s.is_empty()).collect()
}

fn is_var(seg: &str) -> bool {
    seg.len() > 2 && seg.starts_with('{') && seg.ends_with('}')
}

/// The project enclosing `path` (walking up to the filesystem root).
pub fn find(path: &Path) -> Option<Project> {
    let mut d = if path.is_dir() { Some(path) } else { path.parent() };
    while let Some(dir) = d {
        let m = dir.join(DIR).join(FILE);
        if m.is_file() {
            let text = std::fs::read_to_string(&m).ok()?;
            let manifest: Manifest = serde_json::from_str(&text).ok()?;
            if manifest.format > FORMAT {
                log::warn!("{} is a newer organisicator format ({}); ignored", m.display(), manifest.format);
                return None;
            }
            return Some(Project { root: dir.to_path_buf(), manifest });
        }
        d = dir.parent();
    }
    None
}

impl Project {
    pub fn abs(&self, rel: &str) -> PathBuf {
        segs(rel).iter().fold(self.root.clone(), |p, s| p.join(s))
    }

    pub fn instance(&self, id: Option<i64>) -> Option<&Instance> {
        self.manifest.instances.iter().find(|i| Some(i.id) == id)
    }

    /// Enclosing instances, outermost first (`inst` excluded).
    pub fn ancestors(&self, inst: &Instance) -> Vec<&Instance> {
        let mut out = Vec::new();
        let mut p = self.instance(inst.parent);
        while let Some(i) = p {
            if out.len() > 32 {
                break;
            }
            out.push(i);
            p = self.instance(i.parent);
        }
        out.reverse();
        out
    }

    /// The deepest instance whose folder holds `path`.
    pub fn locate(&self, path: &Path) -> Option<&Instance> {
        let rel = path.strip_prefix(&self.root).ok()?;
        let rel: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
        self.manifest
            .instances
            .iter()
            .filter(|i| {
                let ir = segs(&i.rel);
                ir.len() <= rel.len() && ir.iter().zip(&rel).all(|(a, b)| *a == b.as_str())
            })
            .max_by_key(|i| segs(&i.rel).len())
    }

    /// The anatomy folders inside `inst` (not inside a deeper repeating
    /// level), as (definition, absolute path); `None`: the project's own
    /// static folders.
    pub fn folders_of(&self, inst: Option<&Instance>) -> Vec<(&Folder, PathBuf)> {
        let mut out = Vec::new();
        for f in &self.manifest.folders {
            let s = segs(&f.path);
            let ks: Vec<usize> = s.iter().enumerate().filter(|(_, x)| is_var(x)).map(|(k, _)| k).collect();
            match inst {
                None => {
                    if ks.is_empty() {
                        out.push((f, self.abs(&s.join("/"))));
                    }
                }
                Some(i) => {
                    let Some(&k) = ks.last() else { continue };
                    if s[..=k].join("/") != i.decl || k + 1 >= s.len() {
                        continue;
                    }
                    out.push((f, self.abs(&format!("{}/{}", i.rel, s[k + 1..].join("/")))));
                }
            }
        }
        out
    }

    /// The best-scoring folder of `inst`, else of the nearest enclosing
    /// instance that has one, else of the project.
    fn pick(&self, inst: Option<&Instance>, score: impl Fn(&Folder) -> i32) -> Option<PathBuf> {
        let mut chain: Vec<Option<&Instance>> = Vec::new();
        if let Some(i) = inst {
            chain.push(Some(i));
            chain.extend(self.ancestors(i).into_iter().rev().map(Some));
        }
        chain.push(None);
        for level in chain {
            let best = self.folders_of(level).into_iter().map(|(f, p)| (score(f), p)).filter(|(s, _)| *s > 0).max_by_key(|(s, _)| *s);
            if let Some((_, p)) = best {
                return Some(p);
            }
        }
        None
    }

    /// Where an instance's Scrub projects live (the addon's `blender_dir`
    /// rule with Scrub in Blender's place).
    pub fn scrub_dir(&self, inst: Option<&Instance>) -> Option<PathBuf> {
        let found = self.pick(inst, |f| {
            let name = segs(&f.path).last().map(|s| s.to_lowercase()).unwrap_or_default();
            let kind = f.kind.to_lowercase();
            if name == "scrub" || name == "edify" {
                4
            } else if f.exts.to_lowercase().split_whitespace().any(|e| e == "scrub" || e == "edify") {
                3
            } else if kind == "software" || ["software", "scripts", "work"].contains(&name.as_str()) {
                2
            } else {
                0
            }
        })?;
        if found.file_name().is_some_and(|n| ["scrub", "edify"].iter().any(|x| n.to_string_lossy().eq_ignore_ascii_case(x))) {
            return Some(found);
        }
        ["Scrub", "scrub", "Edify", "edify"].iter().map(|c| found.join(c)).find(|p| p.is_dir()).or(Some(found))
    }

    /// Where renders / exports go (the addon's `render_dir`). Scrub has no
    /// export yet; this is where one will default to.
    #[allow(dead_code)]
    pub fn render_dir(&self, inst: Option<&Instance>) -> Option<PathBuf> {
        self.pick(inst, |f| {
            let name = segs(&f.path).last().map(|s| s.to_lowercase()).unwrap_or_default();
            match (f.kind.to_lowercase().as_str(), name.as_str()) {
                ("render" | "renders", _) => 3,
                (_, "renders" | "render") => 2,
                (_, "output" | "out") => 1,
                _ => 0,
            }
        })
    }

    /// "project / sequence / shot" for the title and the browser.
    #[cfg_attr(not(feature = "clips"), allow(dead_code))]
    pub fn label(&self, inst: Option<&Instance>) -> String {
        let mut parts = vec![self.manifest.name.clone()];
        if let Some(i) = inst {
            parts.extend(self.ancestors(i).iter().map(|a| a.value.clone()));
            parts.push(i.value.clone());
        }
        parts.join(" / ")
    }
}

/// `<stem>_edit_v001.scrub` in `dir`, the first version not taken.
pub fn next_version(dir: &Path, stem: &str) -> PathBuf {
    let stem: String = stem.chars().map(|c| if c.is_alphanumeric() || "-_".contains(c) { c } else { '_' }).collect();
    (1..1000).map(|v| dir.join(format!("{stem}_edit_v{v:03}.{}", crate::project::EXT))).find(|p| !p.exists()).unwrap_or_else(|| dir.join(format!("{stem}_edit.{}", crate::project::EXT)))
}

/// Where to save an untitled project whose media is at `media`, and what
/// to call it: the organisicator instance's Scrub folder and a versioned
/// name. `None` outside an organisicator project.
pub fn save_suggestion(media: &Path) -> Option<PathBuf> {
    let p = find(media)?;
    let inst = p.locate(media);
    let dir = p.scrub_dir(inst).unwrap_or_else(|| p.root.clone());
    let stem = inst.map(|i| i.value.clone()).filter(|v| !v.is_empty()).unwrap_or_else(|| {
        if p.manifest.code.is_empty() {
            p.manifest.name.clone()
        } else {
            p.manifest.code.clone()
        }
    });
    Some(next_version(&dir, &stem))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A project laid out like organisicator's "VFX Show" demo, with its
    /// manifest as organisicator writes it.
    fn fixture(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("scrub-org-{tag}-{}", std::process::id())).join("Show");
        let _ = std::fs::remove_dir_all(&root);
        for d in ["sequences/sq010/sh0010/source/plates", "sequences/sq010/sh0010/software/Edify", "sequences/sq010/sh0010/renders", ".organisicator"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        let manifest = r#"{
          "format": 1, "app": "organisicator", "name": "Show", "code": "SHW", "anatomy": "VFX Show",
          "folders": [
            {"path": "sequences", "type": "", "pattern": "", "exts": "", "hint": ""},
            {"path": "sequences/{sequence}", "type": "sequence", "pattern": "sq###", "exts": "", "hint": ""},
            {"path": "sequences/{sequence}/{shot}", "type": "shot", "pattern": "sh####", "exts": "", "hint": ""},
            {"path": "sequences/{sequence}/{shot}/source", "type": "source", "pattern": "", "exts": "exr mov", "hint": "plates"},
            {"path": "sequences/{sequence}/{shot}/software", "type": "software", "pattern": "", "exts": "", "hint": ""},
            {"path": "sequences/{sequence}/{shot}/renders", "type": "render", "pattern": "", "exts": "", "hint": ""}
          ],
          "levels": [],
          "instances": [
            {"id": 1, "parent": null, "decl": "sequences/{sequence}", "value": "sq010", "code": "", "rel": "sequences/sq010"},
            {"id": 2, "parent": 1, "decl": "sequences/{sequence}/{shot}", "value": "sh0010", "code": "", "rel": "sequences/sq010/sh0010"}
          ]
        }"#;
        std::fs::write(root.join(".organisicator/project.json"), manifest).unwrap();
        root
    }

    #[test]
    fn a_clip_finds_its_project_and_shot_and_its_scrub_folder_even_an_old_edify_one() {
        let root = fixture("clip");
        let clip = root.join("sequences/sq010/sh0010/source/plates/bg.mov");
        let p = find(&clip).expect("found");
        assert_eq!(p.manifest.name, "Show");
        let inst = p.locate(&clip).expect("in a shot");
        assert_eq!(inst.value, "sh0010");
        assert_eq!(p.label(Some(inst)), "Show / sq010 / sh0010");
        assert_eq!(p.scrub_dir(Some(inst)), Some(root.join("sequences/sq010/sh0010/software/Edify")));
        assert_eq!(p.render_dir(Some(inst)), Some(root.join("sequences/sq010/sh0010/renders")));
        let s = save_suggestion(&clip).unwrap();
        assert_eq!(s, root.join("sequences/sq010/sh0010/software/Edify/sh0010_edit_v001.scrub"));
        std::fs::write(&s, b"{}").unwrap();
        assert!(save_suggestion(&clip).unwrap().ends_with("sh0010_edit_v002.scrub"), "the next version");
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[test]
    fn outside_a_project_there_is_no_suggestion() {
        assert!(save_suggestion(Path::new("/tmp/nowhere/clip.mov")).is_none());
    }

    #[test]
    fn a_newer_manifest_format_is_ignored() {
        let root = fixture("newer");
        std::fs::write(root.join(".organisicator/project.json"), r#"{"format": 9, "name": "x"}"#).unwrap();
        assert!(find(&root.join("sequences")).is_none());
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }
}
