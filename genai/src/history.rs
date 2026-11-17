//! What made a take: the RECIPE kept beside its result.
//!
//! A take on the node says which file to show; it does not say which
//! prompt, seed and model made it once the pipeline has been edited since.
//! So when a run lands the app writes the steps it ran next to the result
//! (`<results>/<key>/recipe.json`), and the node's history can put them
//! back: "that one, again, but warmer" starts from the settings that made
//! that one.
//!
//! On disk, not in the setup: a recipe is as big as the pipeline, a node
//! keeps every take, and a recipe without its picture is no use to anyone.

use std::path::Path;

use crate::pipeline::{Op, Pipeline, Step};

/// The file a take's recipe lives in, beside `run::META`.
pub const RECIPE: &str = "recipe.json";

/// Record the steps that made the result under `key`. Only beside a
/// finished result, and never over an older recipe: the same key is the
/// same request, so the first one is as true as any.
pub fn save_recipe(dir: &Path, key: &str, steps: &[Step]) -> std::io::Result<()> {
    let d = dir.join(key);
    let f = d.join(RECIPE);
    if key.is_empty() || !d.is_dir() || f.exists() {
        return Ok(());
    }
    let text = serde_json::to_string(steps).map_err(std::io::Error::other)?;
    std::fs::write(f, text)
}

/// The steps that made the result under `key`, if they were recorded.
pub fn load_recipe(dir: &Path, key: &str) -> Option<Vec<Step>> {
    let text = std::fs::read_to_string(dir.join(key).join(RECIPE)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Put a recipe back on a pipeline; the takes are untouched.
///
/// A free-form pipeline takes the steps whole. A ONE-MODEL node keeps its
/// own fixed shape (`fixed` = its catalog id) and takes only what the
/// panel edits: the model step's seed and options, and the typed prompt.
/// A recipe made by another model is refused. Returns whether anything
/// was restored.
pub fn restore(p: &mut Pipeline, recipe: &[Step], fixed: Option<&str>) -> bool {
    let Some(id) = fixed else {
        if recipe.is_empty() {
            return false;
        }
        p.steps = recipe.to_vec();
        for i in 0..p.steps.len() {
            p.sync_ports(i);
        }
        return true;
    };
    let Some((seed, opts)) = recipe.iter().find_map(|s| match &s.op {
        Op::Generate { model, seed, opts } if model == id => Some((*seed, opts.clone())),
        _ => None,
    }) else {
        return false;
    };
    let text = recipe.iter().find_map(|s| match &s.op {
        Op::Prompt { text } => Some(text.clone()),
        _ => None,
    });
    for s in &mut p.steps {
        match &mut s.op {
            Op::Generate { model, seed: sd, opts: o } if model == id => {
                *sd = seed;
                *o = opts.clone();
            }
            Op::Prompt { text: t } => {
                if let Some(text) = &text {
                    *t = text.clone();
                }
            }
            _ => {}
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("wobble-genai-history-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn prompt(p: &mut Pipeline, text: &str) {
        for s in &mut p.steps {
            if let Op::Prompt { text: t } = &mut s.op {
                *t = text.into();
            }
        }
    }

    #[test]
    fn a_recipe_round_trips_beside_its_result_and_is_never_overwritten() {
        let dir = tmp("roundtrip");
        let mut p = Pipeline::default();
        prompt(&mut p, "a moth");
        // No result directory yet: nothing is written.
        save_recipe(&dir, "k", &p.steps).unwrap();
        assert!(load_recipe(&dir, "k").is_none());
        std::fs::create_dir_all(dir.join("k")).unwrap();
        save_recipe(&dir, "k", &p.steps).unwrap();
        assert_eq!(load_recipe(&dir, "k").as_deref(), Some(&p.steps[..]));
        let mut other = p.clone();
        prompt(&mut other, "a lamp");
        save_recipe(&dir, "k", &other.steps).unwrap();
        assert_eq!(load_recipe(&dir, "k").as_deref(), Some(&p.steps[..]), "the first recipe stands");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_free_form_pipeline_takes_the_steps_whole_and_keeps_its_takes() {
        let mut then = Pipeline::default();
        prompt(&mut then, "a moth");
        let mut now = Pipeline::default();
        let extra = now.push(Op::Reference, [0.5, 0.5]);
        now.take = 3;
        assert!(restore(&mut now, &then.steps, None));
        assert_eq!(now.steps, then.steps);
        assert_eq!(now.take, 3);
        assert!(extra >= now.steps.len(), "the step added since is gone");
        assert!(!restore(&mut now, &[], None));
    }

    #[test]
    fn a_one_model_node_keeps_its_shape_and_refuses_another_model_s_recipe() {
        let mut then = Pipeline::for_model("fal/flux-kontext-pro").unwrap();
        prompt(&mut then, "make it night");
        let g = then.model_step().unwrap();
        if let Op::Generate { seed, .. } = &mut then.steps[g].op {
            *seed = 42;
        }
        // The run read a wired prompt: its recipe has no typed one.
        let mut wired = then.clone();
        for s in &mut wired.steps {
            if matches!(s.op, Op::Prompt { .. }) {
                s.op = Op::PromptIn;
            }
        }
        let mut now = Pipeline::for_model("fal/flux-kontext-pro").unwrap();
        prompt(&mut now, "make it day");
        assert!(restore(&mut now, &wired.steps, Some("fal/flux-kontext-pro")));
        assert!(matches!(now.steps[g].op, Op::Generate { seed: 42, .. }));
        assert!(now.steps.iter().any(|s| matches!(&s.op, Op::Prompt { text } if text == "make it day")));
        assert!(restore(&mut now, &then.steps, Some("fal/flux-kontext-pro")));
        assert_eq!(now.steps, then.steps);
        let other = Pipeline::for_model("fal/flux-schnell").unwrap();
        assert!(!restore(&mut now, &other.steps, Some("fal/flux-kontext-pro")));
        assert_eq!(now.steps, then.steps);
    }
}
