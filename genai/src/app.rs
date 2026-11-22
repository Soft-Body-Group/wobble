//! App mode: a pipeline as a FORM.
//!
//! The person who builds a pipeline and the person who runs it are often
//! not the same person, or not the same person in the same mood. An
//! [`AppForm`] picks the few fields worth touching (a prompt, a seed, a
//! scale), gives each a label, and locks the rest: with app mode on, the
//! node's panel is that form and a run button, and the steps cannot be
//! opened.
//!
//! A field names a step by index and what on it (`What`). Indices are the
//! pipeline's ids, and they slide when a step is removed, so a form is
//! [`AppForm::prune`]d against the pipeline whenever it is shown: a field
//! whose step no longer has that thing is dropped, never guessed at. With
//! app mode ON the steps cannot change, so a form in use never moves.
//!
//! Nothing here runs or bills: a form edits the same authored values the
//! full panel does, and the run goes through the same plan and quote.

use serde::{Deserialize, Serialize};

use crate::catalog::{self, Task};
use crate::pipeline::{Op, Pipeline};

/// Which value of a step a field edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum What {
    /// A Prompt's text, or a Text Template's.
    Text,
    /// A model step's (or ComfyUI step's) seed.
    Seed,
    /// An upscaler's factor, or an outpaint's canvas growth.
    Scale,
    /// A text-to-image step's megapixels.
    Megapixels,
    /// An image-to-video step's length.
    Seconds,
    /// An Input Clip's length.
    ClipSeconds,
}

/// One row of the form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub step: usize,
    pub what: What,
    /// What the form calls it. Empty = the default label.
    #[serde(default)]
    pub label: String,
}

/// A pipeline's form, and whether the node is showing it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AppForm {
    /// App mode: the panel is the form, and the steps are locked.
    #[serde(default)]
    pub on: bool,
    /// The form's heading. Empty = none.
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub fields: Vec<Field>,
}

/// A number field's range, and whether it is whole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    pub lo: f32,
    pub hi: f32,
    pub whole: bool,
}

/// Does step `i` of `p` have a `what` to edit?
pub fn has(p: &Pipeline, i: usize, what: What) -> bool {
    let Some(s) = p.steps.get(i) else { return false };
    match (&s.op, what) {
        (Op::Prompt { .. } | Op::Template { .. }, What::Text) => true,
        (Op::Generate { .. } | Op::Comfy { .. }, What::Seed) => true,
        (Op::Input { clip: true, .. }, What::ClipSeconds) => true,
        (Op::Generate { model, .. }, w) => match (catalog::get(model), w) {
            (Some(m), What::Scale) => {
                matches!(m.task, Task::Upscale | Task::VideoUpscale | Task::Outpaint) && m.dialect.scale_range().is_some()
            }
            (Some(m), What::Megapixels) => m.task == Task::TextToImage,
            (Some(m), What::Seconds) => m.dialect.video_range().is_some(),
            _ => false,
        },
        _ => false,
    }
}

/// Everything in `p` a form could offer, in plan order (steps that lead
/// nowhere are not offered: their values change nothing).
pub fn candidates(p: &Pipeline) -> Vec<(usize, What)> {
    let Some(order) = p.output().and_then(|o| p.plan(o).ok()) else { return vec![] };
    let all = [What::Text, What::Seed, What::Scale, What::Megapixels, What::Seconds, What::ClipSeconds];
    order.into_iter().flat_map(|i| all.into_iter().filter(move |&w| has(p, i, w)).map(move |w| (i, w))).collect()
}

/// What a field is called when it has no label of its own.
pub fn default_label(p: &Pipeline, i: usize, what: What) -> String {
    let step = p.steps.get(i).map(|s| s.op.name()).unwrap_or("step");
    match what {
        What::Text => match p.steps.get(i).map(|s| &s.op) {
            Some(Op::Template { .. }) => "template".into(),
            _ => "prompt".into(),
        },
        What::Seed => format!("{step} seed"),
        What::Scale => format!("{step} scale"),
        What::Megapixels => "megapixels".into(),
        What::Seconds => "seconds".into(),
        What::ClipSeconds => "clip length (s)".into(),
    }
}

/// A number field's range (`None` for text).
pub fn range(p: &Pipeline, i: usize, what: What) -> Option<Range> {
    let model = match p.steps.get(i).map(|s| &s.op) {
        Some(Op::Generate { model, .. }) => catalog::get(model),
        _ => None,
    };
    Some(match what {
        What::Text => return None,
        What::Seed => Range { lo: 0.0, hi: 999_999.0, whole: true },
        What::Scale => {
            let (lo, hi) = model.and_then(|m| m.dialect.scale_range())?;
            Range { lo, hi, whole: false }
        }
        What::Megapixels => {
            let (lo, hi) = model?.dialect.mp_range();
            let lo = lo.max(0.25);
            Range { lo, hi: hi.min(2.0).max(lo), whole: false }
        }
        What::Seconds => {
            let (lo, hi, _) = model?.dialect.video_range()?;
            Range { lo: lo as f32, hi: hi as f32, whole: true }
        }
        What::ClipSeconds => Range { lo: 0.5, hi: 30.0, whole: false },
    })
}

/// A text field's value.
pub fn text_mut(p: &mut Pipeline, i: usize) -> Option<&mut String> {
    match p.steps.get_mut(i).map(|s| &mut s.op) {
        Some(Op::Prompt { text } | Op::Template { text }) => Some(text),
        _ => None,
    }
}

/// A number field's value.
pub fn number(p: &Pipeline, i: usize, what: What) -> Option<f32> {
    match (p.steps.get(i).map(|s| &s.op)?, what) {
        (Op::Generate { seed, .. } | Op::Comfy { seed, .. }, What::Seed) => Some(*seed as f32),
        (Op::Generate { opts, .. }, What::Scale) => Some(opts.scale),
        (Op::Generate { opts, .. }, What::Megapixels) => Some(opts.megapixels),
        (Op::Generate { opts, .. }, What::Seconds) => Some(opts.seconds as f32),
        (Op::Input { clip: true, seconds, .. }, What::ClipSeconds) => Some(*seconds),
        _ => None,
    }
}

/// Set a number field, clamped to its range. Returns whether it changed.
pub fn set_number(p: &mut Pipeline, i: usize, what: What, v: f32) -> bool {
    let Some(r) = range(p, i, what) else { return false };
    let v = v.clamp(r.lo, r.hi);
    if what == What::ClipSeconds && has(p, i, what) {
        // Every clip a run sends covers the same frames.
        return p.set_clip_seconds(v);
    }
    let Some(s) = p.steps.get_mut(i) else { return false };
    match (&mut s.op, what) {
        (Op::Generate { seed, .. } | Op::Comfy { seed, .. }, What::Seed) => replace(seed, v.round() as u32),
        (Op::Generate { opts, .. }, What::Scale) => replace(&mut opts.scale, v),
        (Op::Generate { opts, .. }, What::Megapixels) => replace(&mut opts.megapixels, v),
        (Op::Generate { opts, .. }, What::Seconds) => replace(&mut opts.seconds, v.round() as u32),
        _ => false,
    }
}

fn replace<T: PartialEq>(slot: &mut T, v: T) -> bool {
    let changed = *slot != v;
    *slot = v;
    changed
}

impl AppForm {
    /// Not on, no title, no fields: nothing to save.
    pub fn is_empty(&self) -> bool {
        !self.on && self.title.is_empty() && self.fields.is_empty()
    }

    /// Is `(step, what)` on the form?
    pub fn shows(&self, step: usize, what: What) -> bool {
        self.fields.iter().any(|f| f.step == step && f.what == what)
    }

    /// Put `(step, what)` on the form (at the end) or take it off.
    pub fn toggle(&mut self, step: usize, what: What, on: bool) {
        self.fields.retain(|f| !(f.step == step && f.what == what));
        if on {
            self.fields.push(Field { step, what, label: String::new() });
        }
    }

    /// Drop fields whose step no longer has that value (a step removed, a
    /// model changed). Returns whether anything went.
    pub fn prune(&mut self, p: &Pipeline) -> bool {
        let n = self.fields.len();
        self.fields.retain(|f| has(p, f.step, f.what));
        self.fields.len() != n
    }

    /// The form a pipeline gets when asked for one with no thought: every
    /// prompt, and the seed of the model feeding Output.
    pub fn suggested(p: &Pipeline) -> Self {
        let mut form = AppForm::default();
        let out_model = p.output().and_then(|o| p.steps[o].inputs.first().copied().flatten());
        for (i, w) in candidates(p) {
            let prompt = w == What::Text && matches!(p.steps[i].op, Op::Prompt { .. });
            if prompt || (w == What::Seed && Some(i) == out_model) {
                form.toggle(i, w, true);
            }
        }
        form
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Opts;

    #[test]
    fn candidates_follow_the_steps_on_the_way_out() {
        let p = Pipeline::default(); // prompt -> schnell -> out
        assert_eq!(candidates(&p), [(0, What::Text), (1, What::Seed), (1, What::Megapixels)]);
        let up = Pipeline::for_model("fal/topaz-upscale").unwrap();
        let g = up.model_step().unwrap();
        assert!(candidates(&up).contains(&(g, What::Scale)));
        // Recraft picks its own size: no scale to offer.
        let rc = Pipeline::for_model("fal/recraft-crisp-upscale").unwrap();
        assert!(!candidates(&rc).iter().any(|c| c.1 == What::Scale));
        // A step that leads nowhere is not offered.
        let mut p = Pipeline::default();
        p.push(Op::Prompt { text: "stray".into() }, [0.0, 0.0]);
        assert_eq!(candidates(&p).len(), 3);
    }

    #[test]
    fn the_suggested_form_is_the_prompts_and_the_last_seed() {
        let p = crate::templates::BUILTIN[0].pipeline(); // prompt -> schnell -> topaz -> out
        let f = AppForm::suggested(&p);
        let topaz = p.steps.iter().position(|s| matches!(&s.op, Op::Generate { model, .. } if model == "fal/topaz-upscale")).unwrap();
        assert_eq!(f.fields.len(), 2);
        assert!(f.shows(0, What::Text) && f.shows(topaz, What::Seed));
        assert!(!f.on, "suggesting a form does not lock anything");
    }

    #[test]
    fn numbers_clamp_to_the_model_s_range_and_seeds_stay_whole() {
        let mut p = Pipeline::default();
        assert!(set_number(&mut p, 1, What::Seed, 41.6));
        assert_eq!(number(&p, 1, What::Seed), Some(42.0));
        assert!(set_number(&mut p, 1, What::Megapixels, 99.0));
        assert_eq!(number(&p, 1, What::Megapixels), Some(2.0));
        assert!(!set_number(&mut p, 1, What::Megapixels, 99.0), "no change the second time");
        assert!(!set_number(&mut p, 0, What::Seed, 3.0), "a prompt has no seed");
        *text_mut(&mut p, 0).unwrap() = "a moth".into();
        assert!(matches!(&p.steps[0].op, Op::Prompt { text } if text == "a moth"));
        assert!(text_mut(&mut p, 1).is_none());
    }

    #[test]
    fn a_form_is_pruned_when_the_pipeline_moves_under_it() {
        let mut p = Pipeline::default();
        let mut f = AppForm::suggested(&p);
        f.toggle(1, What::Megapixels, true);
        assert_eq!(f.fields.len(), 3);
        assert!(!f.prune(&p));
        // The model becomes an editor: megapixels are gone, the rest stay.
        p.set_model(1, "fal/flux-kontext-pro");
        assert!(f.prune(&p));
        assert!(f.shows(0, What::Text) && f.shows(1, What::Seed) && !f.shows(1, What::Megapixels));
        // The prompt step is removed: every index slides, and what no
        // longer matches is dropped rather than re-pointed.
        p.remove(0);
        f.prune(&p);
        assert!(f.fields.iter().all(|x| has(&p, x.step, x.what)));
    }

    #[test]
    fn an_empty_form_is_not_saved_and_a_form_round_trips() {
        assert_eq!(serde_json::to_string(&AppForm::default()).unwrap(), r#"{"on":false,"title":"","fields":[]}"#);
        assert!(AppForm::default().is_empty());
        let mut f = AppForm::suggested(&Pipeline::default());
        f.on = true;
        f.title = "Moth maker".into();
        f.fields[0].label = "what should it be?".into();
        let back: AppForm = serde_json::from_str(&serde_json::to_string(&f).unwrap()).unwrap();
        assert_eq!(back, f);
        assert_eq!(serde_json::from_str::<AppForm>("{}").unwrap(), AppForm::default());
        let _ = Opts::default();
    }
}
