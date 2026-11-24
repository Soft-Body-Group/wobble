//! Templates: whole pipelines, ready to drop on a GenAI Pipeline node.
//!
//! Two shelves, one shape:
//!
//! - **Built in** ([`BUILTIN`]): the workflows people reach for first, each
//!   a few catalog models wired the way they are meant to be used. Code,
//!   not files, so they move with the catalog and a test can hold every
//!   one of them to "plans, and every port is fed".
//! - **Yours** ([`user_dir`], `~/.wobble/genai-templates/*.json`): any
//!   pipeline saved from the panel. One small JSON file each, so SHARING
//!   one is sending the file: drop it in the folder and it is on the
//!   shelf. A file names catalog ids and nothing else (no keys, no URLs,
//!   no takes), exactly like a setup.
//!
//! A template is only ever STEPS. Applying one replaces a pipeline's steps
//! and leaves its takes alone, and runs nothing.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::catalog;
use crate::pipeline::{Op, Opts, Pipeline, Step};

/// A built-in template.
pub struct Template {
    /// What the shelf, and the add menu, call it.
    pub name: &'static str,
    /// One line on what it is for.
    pub about: &'static str,
    build: fn() -> Pipeline,
}

impl Template {
    pub fn pipeline(&self) -> Pipeline {
        let mut p = (self.build)();
        p.normalize();
        p
    }

    /// The catalog models it runs, in step order.
    pub fn models(&self) -> Vec<&'static catalog::RemoteModel> {
        models_of(&self.pipeline().steps)
    }
}

/// The catalog models a set of steps runs (unknown ids are skipped).
pub fn models_of(steps: &[Step]) -> Vec<&'static catalog::RemoteModel> {
    steps
        .iter()
        .filter_map(|s| match &s.op {
            Op::Generate { model, .. } => catalog::get(model),
            _ => None,
        })
        .collect()
}

/// A chain builder: sources across the top, models down the middle.
struct B {
    p: Pipeline,
    row: f32,
}

impl B {
    fn new() -> Self {
        // Built with the old source steps (Reference, Matte, Second…):
        // `Template::pipeline` normalizes them into the node's Inputs.
        B { p: Pipeline { steps: vec![], takes: vec![], take: 0, ins: vec![], made: vec![] }, row: -0.2 }
    }
    fn src(&mut self, op: Op, x: f32) -> usize {
        self.p.push(op, [x, -0.6])
    }
    fn prompt(&mut self, text: &str, x: f32) -> usize {
        self.src(Op::Prompt { text: text.into() }, x)
    }
    /// A model step fed by `ins` (one per port, in port order).
    fn model(&mut self, id: &str, ins: &[usize]) -> usize {
        self.model_with(id, Opts::default(), ins)
    }
    fn model_with(&mut self, id: &str, opts: Opts, ins: &[usize]) -> usize {
        let g = self.p.push(Op::Generate { model: id.into(), seed: 1, opts }, [0.0, self.row]);
        self.row += 0.3;
        for (port, &from) in ins.iter().enumerate() {
            self.p.connect(from, g, port);
        }
        g
    }
    fn out(mut self, from: usize) -> Pipeline {
        let o = self.p.push(Op::Output, [0.0, self.row.max(0.55)]);
        self.p.connect(from, o, 0);
        self.p
    }
}

pub const BUILTIN: &[Template] = &[
    Template {
        name: "Concept, upscaled",
        about: "a prompt to a picture with FLUX schnell, then Topaz to twice the size",
        build: || {
            let mut b = B::new();
            let t = b.prompt("", -0.4);
            let g = b.model("fal/flux-schnell", &[t]);
            let u = b.model("fal/topaz-upscale", &[g]);
            b.out(u)
        },
    },
    Template {
        name: "Clean plate",
        about: "repaint the matte's white from the plate around it (FLUX Fill): remove a sign, a rig, a person",
        build: || {
            let mut b = B::new();
            let t = b.prompt("empty background, matching the surroundings", -0.5);
            let i = b.src(Op::Reference, 0.0);
            let m = b.src(Op::Matte, 0.5);
            let g = b.model("fal/flux-fill-pro", &[t, i, m]);
            b.out(g)
        },
    },
    Template {
        name: "Restyle the plate",
        about: "edit the plate by instruction with Kontext (\"make it night\", \"heavy snow\")",
        build: || {
            let mut b = B::new();
            let t = b.prompt("", -0.4);
            let i = b.src(Op::Reference, 0.4);
            let g = b.model("fal/flux-kontext-pro", &[t, i]);
            b.out(g)
        },
    },
    Template {
        name: "New background, upscaled",
        about: "keep the subject, paint a new background from the prompt (Bria), then Topaz to twice the size",
        build: || {
            let mut b = B::new();
            let t = b.prompt("", -0.4);
            let i = b.src(Op::Reference, 0.4);
            let g = b.model("fal/bria-background-replace", &[t, i]);
            let u = b.model("fal/topaz-upscale", &[g]);
            b.out(u)
        },
    },
    Template {
        name: "Outpaint the plate",
        about: "grow the canvas 1.5x around the plate and paint what was out of frame (Bria Expand)",
        build: || {
            let mut b = B::new();
            let t = b.prompt("", -0.4);
            let i = b.src(Op::Reference, 0.4);
            let g = b.model_with("fal/bria-expand", Opts { scale: 1.5, ..Opts::default() }, &[t, i]);
            b.out(g)
        },
    },
    Template {
        name: "Match a look",
        about: "give the plate the palette and grade of the picture on the second input (Colour Transfer)",
        build: || {
            let mut b = B::new();
            let t = b.prompt("", -0.5);
            let i = b.src(Op::Reference, 0.0);
            let s = b.src(Op::Second, 0.5);
            let g = b.model("fal/colour-transfer", &[t, i, s]);
            b.out(g)
        },
    },
    Template {
        name: "Relight from a prompt",
        about: "relight the plate from a prompt with IC-Light (\"warm sunset light from the left\")",
        build: || {
            let mut b = B::new();
            let t = b.prompt("warm sunset light from the left", -0.4);
            let i = b.src(Op::Reference, 0.4);
            let g = b.model("fal/iclight-v2", &[t, i]);
            b.out(g)
        },
    },
    Template {
        name: "Cut out, on a new background",
        about: "cut the subject out (BiRefNet), then set it on a background painted from the prompt (Bria)",
        build: || {
            let mut b = B::new();
            let i = b.src(Op::Reference, -0.4);
            let t = b.prompt("", 0.4);
            let c = b.model("fal/birefnet-v2", &[i]);
            let g = b.model("fal/bria-background-replace", &[t, c]);
            b.out(g)
        },
    },
    Template {
        name: "Concept to clip",
        about: "a prompt to a still (Nano Banana 2), then that still animated by the same prompt (Seedance)",
        build: || {
            let mut b = B::new();
            let t = b.prompt("", -0.4);
            let g = b.model("fal/nano-banana-2", &[t]);
            let v = b.model("fal/seedance-2-fast-i2v", &[t, g]);
            b.out(v)
        },
    },
    Template {
        name: "Animate the plate",
        about: "the plate at the playhead as the first frame of a clip, moved as the prompt says (Seedance)",
        build: || {
            let mut b = B::new();
            let t = b.prompt("", -0.4);
            let i = b.src(Op::Reference, 0.4);
            let v = b.model("fal/seedance-2-fast-i2v", &[t, i]);
            b.out(v)
        },
    },
    Template {
        name: "Remove from the clip",
        about: "fill the matte's white over the input clip (LTX inpaint): object removal on video",
        build: || {
            let mut b = B::new();
            let t = b.prompt("empty background, matching the surroundings", -0.5);
            let c = b.src(Op::ClipIn { seconds: 5.0 }, 0.0);
            let m = b.src(Op::MatteClip, 0.5);
            let v = b.model("fal/ltx-2.3-inpaint", &[t, c, m]);
            b.out(v)
        },
    },
    Template {
        name: "Restore the clip",
        about: "restore and upscale the input clip with SeedVR2: soft, noisy or generated footage",
        build: || {
            let mut b = B::new();
            let c = b.src(Op::ClipIn { seconds: 5.0 }, 0.0);
            let v = b.model("fal/seedvr2-video-upscale", &[c]);
            b.out(v)
        },
    },
    Template {
        name: "Enhanced concept",
        about: "a short prompt, written out in full by the Prompt Enhancer, then FLUX schnell",
        build: || {
            let mut b = B::new();
            let t = b.prompt("", -0.4);
            let e = b.model("fal/prompt-enhance", &[t]);
            let g = b.model("fal/flux-schnell", &[e]);
            b.out(g)
        },
    },
    Template {
        name: "Describe and reimagine",
        about: "the plate described in words, a style of yours added ({a}, {b}), and a new picture made from that",
        build: || {
            let mut b = B::new();
            let ask = b.prompt("", -0.6);
            let i = b.src(Op::Reference, -0.2);
            let style = b.prompt("as a watercolour painting", 0.5);
            let d = b.model("fal/describe", &[ask, i]);
            let t = b.p.push(Op::Template { text: "{a}, {b}".into() }, [0.3, b.row]);
            b.row += 0.3;
            b.p.connect(d, t, 0);
            b.p.connect(style, t, 1);
            let g = b.model("fal/flux-schnell", &[t]);
            b.out(g)
        },
    },
];

// ------------------------------------------------------------ yours

/// Marks a file as one of ours, and versions the shape.
const KIND: &str = "wobble-genai-template/1";

/// A saved template, as it sits on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Saved {
    pub kind: String,
    pub name: String,
    #[serde(default)]
    pub about: String,
    pub steps: Vec<Step>,
    /// The pipeline's app form, when it had one: a template shared as an
    /// app opens as that form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app: Option<crate::AppForm>,
}

/// `$WOBBLE_GENAI_TEMPLATES`, else `~/.wobble/genai-templates`.
pub fn user_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("WOBBLE_GENAI_TEMPLATES") {
        return PathBuf::from(p);
    }
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".wobble/genai-templates")
}

/// A file name for a template name: its letters and digits, the rest `-`.
pub fn file_name(name: &str) -> String {
    let mut s: String = name
        .trim()
        .chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect();
    while s.contains("--") {
        s = s.replace("--", "-");
    }
    let s = s.trim_matches('-');
    format!("{}.json", if s.is_empty() { "template" } else { s })
}

/// Save `steps` as a template in `dir` (made if missing), over any of the
/// same name. Returns the file.
pub fn save(dir: &Path, name: &str, about: &str, steps: &[Step]) -> std::io::Result<PathBuf> {
    save_with(dir, name, about, steps, None)
}

/// [`save`], with the pipeline's app form riding along.
pub fn save_with(
    dir: &Path,
    name: &str,
    about: &str,
    steps: &[Step],
    app: Option<&crate::AppForm>,
) -> std::io::Result<PathBuf> {
    if name.trim().is_empty() {
        return Err(std::io::Error::other("a template needs a name"));
    }
    std::fs::create_dir_all(dir)?;
    let t = Saved {
        kind: KIND.into(),
        name: name.trim().into(),
        about: about.trim().into(),
        steps: steps.to_vec(),
        app: app.filter(|a| !a.is_empty()).cloned(),
    };
    let path = dir.join(file_name(name));
    let text = serde_json::to_string_pretty(&t).map_err(std::io::Error::other)?;
    std::fs::write(&path, text)?;
    Ok(path)
}

/// Read one template file. Its steps come back with their ports synced, so
/// a file written by a build with a different catalog still wires up as
/// far as this build's ports allow.
pub fn load(path: &Path) -> Result<Saved, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut t: Saved = serde_json::from_str(&text).map_err(|e| format!("not a template: {e}"))?;
    if t.kind != KIND {
        return Err(format!("not a wobble GenAI template ({})", t.kind));
    }
    let mut p = Pipeline { steps: std::mem::take(&mut t.steps), takes: vec![], take: 0, ins: vec![], made: vec![] };
    for i in 0..p.steps.len() {
        p.sync_ports(i);
    }
    if let Some(a) = &mut t.app {
        a.prune(&p);
    }
    t.steps = p.steps;
    Ok(t)
}

/// Every template in `dir`, by name. Files that are not templates are
/// skipped: it is a folder people drop things into.
pub fn list(dir: &Path) -> Vec<(PathBuf, Saved)> {
    let Ok(rd) = std::fs::read_dir(dir) else { return vec![] };
    let mut v: Vec<(PathBuf, Saved)> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("json")))
        .filter_map(|p| load(&p).ok().map(|t| (p, t)))
        .collect();
    v.sort_by(|a, b| a.1.name.to_lowercase().cmp(&b.1.name.to_lowercase()));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_built_in_plans_with_every_port_fed_and_known_models() {
        let mut names: Vec<_> = BUILTIN.iter().map(|t| t.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), BUILTIN.len(), "names are unique");
        for t in BUILTIN {
            let p = t.pipeline();
            let out = p.output().unwrap_or_else(|| panic!("{}: no Output", t.name));
            let order = p.plan(out).unwrap_or_else(|e| panic!("{}: {e}", t.name));
            // Every step is on the way out, but an Input nothing reads (the
            // node keeps its inputs whether a template uses them or not).
            let used = p.steps.iter().filter(|s| !matches!(s.op, Op::Input { .. })).count() + p.reads().len();
            assert_eq!(order.len(), used, "{}: a step leads nowhere", t.name);
            for s in &p.steps {
                assert!(s.inputs.iter().all(|i| i.is_some()), "{}: {} has an unfed port", t.name, s.op.name());
                if let Op::Generate { model, .. } = &s.op {
                    assert!(catalog::get(model).is_some(), "{}: unknown model {model}", t.name);
                }
            }
            assert!(!t.models().is_empty() && !t.about.is_empty(), "{}", t.name);
        }
    }

    #[test]
    fn a_saved_template_round_trips_and_strangers_in_the_folder_are_skipped() {
        let dir = std::env::temp_dir().join(format!("wobble-genai-templates-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = BUILTIN[1].pipeline();
        let f = save(&dir, "  My clean plate! ", "for the bridge shots", &p.steps).unwrap();
        assert_eq!(f.file_name().unwrap(), "my-clean-plate.json");
        std::fs::write(dir.join("notes.json"), "{\"hello\": 1}").unwrap();
        std::fs::write(dir.join("readme.txt"), "hi").unwrap();
        let all = list(&dir);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].1.name, "My clean plate!");
        assert_eq!(all[0].1.steps, p.steps);
        // No takes, keys or URLs ride along: a template is steps.
        let text = std::fs::read_to_string(&f).unwrap();
        assert!(!text.contains("takes") && !text.contains("http"), "{text}");
        assert!(save(&dir, "  ", "", &p.steps).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_template_saved_from_an_app_opens_as_that_app() {
        let dir = std::env::temp_dir().join(format!("wobble-genai-templates-app-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let p = BUILTIN[0].pipeline();
        let mut form = crate::AppForm::suggested(&p);
        form.on = true;
        form.title = "Concept".into();
        let f = save_with(&dir, "concept app", "", &p.steps, Some(&form)).unwrap();
        assert_eq!(load(&f).unwrap().app, Some(form));
        // No form, no field in the file.
        let f = save(&dir, "plain", "", &p.steps).unwrap();
        assert!(!std::fs::read_to_string(&f).unwrap().contains("\"app\""));
        assert_eq!(load(&f).unwrap().app, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_template_with_a_model_this_build_lacks_still_loads() {
        let dir = std::env::temp_dir().join(format!("wobble-genai-templates-x-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut p = BUILTIN[0].pipeline();
        if let Some(Op::Generate { model, .. }) = p.steps.iter_mut().map(|s| &mut s.op).find(|o| o.is_remote()) {
            *model = "fal/from-the-future".into();
        }
        let f = save(&dir, "future", "", &p.steps).unwrap();
        let t = load(&f).unwrap();
        assert_eq!(t.steps.len(), p.steps.len());
        assert!(models_of(&t.steps).len() < models_of(&BUILTIN[0].pipeline().steps).len());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
