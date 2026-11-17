//! Variations: one run of a node that makes SEVERAL takes.
//!
//! A [`Vary`] sits beside the pipeline and says how to multiply it, along
//! four axes that multiply each other:
//!
//! - **models** (fan-out): the same prompt, seed and inputs through other
//!   models as well, so their results land side by side in the history;
//! - **prompts**: a list, each run in place of the pipeline's own prompt;
//! - **seeds**: so many more seeds after the authored one;
//! - **frames**: the comp frames to capture the inputs at, instead of the
//!   playhead — which is how a pipeline runs over a list of IMAGES: a Load
//!   of ten stills is ten frames.
//!
//! [`Vary::expand`] turns a pipeline into its variants (models, prompts,
//! seeds), the pipeline itself first; the frames are the host's to walk,
//! because only it can render an input. Each variant is an ordinary pipeline: planned, quoted, cached and
//! licensed exactly like a single run, which is the point — nothing about
//! billing or caching knows a batch exists (`run::start_many` just walks
//! them in turn). The quote the user confirms is the sum.

use serde::{Deserialize, Serialize};

use crate::catalog::{self, RemoteModel};
use crate::pipeline::{Op, Pipeline};

/// How a node's run is multiplied. Empty = one run, as ever.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Vary {
    /// Catalog ids ALSO run in place of the pipeline's fan-out step
    /// ([`fan_step`]). Ids this build does not know, a model of another
    /// task, and the step's own model are skipped, not errors: a setup
    /// from a newer build still runs what this one can.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub models: Vec<String>,
    /// Prompts run in place of the pipeline's prompt step, one variant
    /// each. Blank entries are dropped. Empty = the pipeline's own prompt.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prompts: Vec<String>,
    /// How many MORE seeds to run after the authored one, each the next
    /// in the panel's "new seed" sequence.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub seeds: u32,
    /// The comp frames to run at, as typed: `1001, 1010-1050x10`. Empty =
    /// the playhead. Text, so a half-typed list is kept and reported
    /// (`ticks`) rather than silently dropped.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub frames: String,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// The most runs one click may ask for. A list pasted by accident should
/// meet a sentence, not a quote for four hundred pictures.
pub const MAX_RUNS: usize = 64;

/// The seed after `seed`: the panel's "new seed" step.
pub fn next_seed(seed: u32) -> u32 {
    seed.wrapping_mul(1_103_515_245).wrapping_add(12_345) % 1_000_000
}

/// A typed frame list: numbers and ranges, separated by commas or spaces.
/// `a-b` is every frame from a to b, `a-bxN` every Nth. In order, each
/// frame once.
pub fn parse_frames(text: &str) -> Result<Vec<i64>, String> {
    let mut out: Vec<i64> = vec![];
    for tok in text.split(|c: char| c == ',' || c.is_whitespace()).filter(|t| !t.is_empty()) {
        let bad = || format!("\"{tok}\" is not a frame or a range (1001, 1010-1050, 1010-1050x10)");
        let (range, step) = match tok.split_once(['x', 'X']) {
            Some((r, s)) => (r, s.parse::<i64>().ok().filter(|&s| s > 0).ok_or_else(bad)?),
            None => (tok, 1),
        };
        // A leading '-' is a sign, not a range.
        let cut = range.char_indices().skip(1).find(|&(_, c)| c == '-').map(|(i, _)| i);
        let (a, b) = match cut {
            Some(i) => {
                (range[..i].parse::<i64>().map_err(|_| bad())?, range[i + 1..].parse::<i64>().map_err(|_| bad())?)
            }
            None => {
                let a = range.parse::<i64>().map_err(|_| bad())?;
                (a, a)
            }
        };
        if b < a {
            return Err(format!("\"{tok}\" runs backwards"));
        }
        let mut f = a;
        while f <= b {
            if !out.contains(&f) {
                out.push(f);
            }
            if out.len() > MAX_RUNS {
                return Err(format!("more than {MAX_RUNS} frames"));
            }
            f += step;
        }
    }
    Ok(out)
}

/// The step whose text a prompt list replaces: the first text source on
/// the way to Output (a typed Prompt, or the node's wired one).
pub fn prompt_step(p: &Pipeline) -> Option<usize> {
    let order = p.plan(p.output()?).ok()?;
    order.into_iter().find(|&i| matches!(p.steps[i].op, Op::Prompt { .. } | Op::PromptIn))
}

/// The step a fan-out swaps models on: the model feeding Output, else the
/// first model step.
pub fn fan_step(p: &Pipeline) -> Option<usize> {
    let fed = p
        .output()
        .and_then(|o| p.steps[o].inputs.first().copied().flatten())
        .filter(|&i| matches!(p.steps[i].op, Op::Generate { .. }));
    fed.or_else(|| p.model_step())
}

/// The fan-out step's own model.
pub fn fan_model(p: &Pipeline) -> Option<&'static RemoteModel> {
    match &p.steps[fan_step(p)?].op {
        Op::Generate { model, .. } => catalog::get(model),
        _ => None,
    }
}

/// The models that can stand in for the fan-out step: the same task (so
/// the same ports, and a result that means the same thing), not itself.
pub fn alternatives(p: &Pipeline) -> Vec<&'static RemoteModel> {
    let Some(own) = fan_model(p) else { return vec![] };
    catalog::for_task(own.task).filter(|m| m.id != own.id).collect()
}

impl Vary {
    pub fn is_empty(&self) -> bool {
        self.models.is_empty() && self.prompts.is_empty() && self.seeds == 0 && self.frames.trim().is_empty()
    }

    /// The frames to run at; empty = the playhead.
    pub fn ticks(&self) -> Result<Vec<i64>, String> {
        parse_frames(&self.frames)
    }

    /// The prompts that would run on `p`: the list's non-blank entries,
    /// when `p` has a prompt to replace.
    pub fn prompts_for(&self, p: &Pipeline) -> Vec<&str> {
        if prompt_step(p).is_none() {
            return vec![];
        }
        self.prompts.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect()
    }

    /// The fan-out models that would actually run on `p`, in list order,
    /// each once.
    pub fn models_for(&self, p: &Pipeline) -> Vec<&'static RemoteModel> {
        let alts = alternatives(p);
        let mut out: Vec<&'static RemoteModel> = vec![];
        for id in &self.models {
            if let Some(m) = alts.iter().find(|m| m.id == id) {
                if !out.iter().any(|o| o.id == m.id) {
                    out.push(m);
                }
            }
        }
        out
    }

    /// Every pipeline a run walks at ONE frame: prompts, then seeds, then
    /// models, innermost last — so the takes land grouped the way a sheet
    /// is read (a prompt's seeds together, a seed's models together). With
    /// no prompt list the pipeline's own prompt runs; `p` itself is first.
    pub fn expand(&self, p: &Pipeline) -> Vec<Pipeline> {
        let mut by_prompt = vec![];
        let prompts = self.prompts_for(p);
        if prompts.is_empty() {
            by_prompt.push(p.clone());
        }
        for text in prompts {
            let mut v = p.clone();
            // Checked by `prompts_for`.
            let i = prompt_step(p).unwrap();
            v.steps[i].op = Op::Prompt { text: text.to_string() };
            by_prompt.push(v);
        }
        let mut by_seed = vec![];
        for base in by_prompt {
            let mut v = base;
            for k in 0..=self.seeds {
                if k > 0 {
                    for s in &mut v.steps {
                        if let Op::Generate { seed, .. } | Op::Comfy { seed, .. } = &mut s.op {
                            *seed = next_seed(*seed);
                        }
                    }
                }
                by_seed.push(v.clone());
            }
        }
        let models = self.models_for(p);
        let step = fan_step(p);
        let mut out = vec![];
        for base in by_seed {
            out.push(base.clone());
            for m in &models {
                let mut v = base.clone();
                v.set_model(step.unwrap(), m.id);
                out.push(v);
            }
        }
        out
    }

    /// How many takes a run of `p` makes at one frame.
    pub fn count(&self, p: &Pipeline) -> usize {
        self.prompts_for(p).len().max(1) * (1 + self.seeds as usize) * (1 + self.models_for(p).len())
    }

    /// Would every variant's extra model be free for commercial use? (The
    /// pipeline's own models are `Pipeline::all_commercial`'s to answer.)
    pub fn all_commercial(&self, p: &Pipeline) -> bool {
        self.models_for(p).iter().all(|m| m.commercial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Task;

    fn ids(v: &[Pipeline]) -> Vec<String> {
        v.iter()
            .map(|p| match &p.steps[fan_step(p).unwrap()].op {
                Op::Generate { model, .. } => model.clone(),
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn nothing_to_vary_is_the_pipeline_itself() {
        let p = Pipeline::default();
        assert_eq!(Vary::default().expand(&p), vec![p.clone()]);
        assert_eq!(Vary::default().count(&p), 1);
    }

    #[test]
    fn a_fan_out_swaps_the_model_and_keeps_every_wire() {
        let p = Pipeline::default(); // prompt -> schnell -> out
        let v = Vary { models: vec!["fal/flux-dev".into(), "fal/nano-banana-2".into()], ..Default::default() };
        let all = v.expand(&p);
        assert_eq!(ids(&all), ["fal/flux-schnell", "fal/flux-dev", "fal/nano-banana-2"]);
        for x in &all {
            let out = x.output().unwrap();
            assert_eq!(x.plan(out).unwrap().len(), x.steps.len(), "still wired end to end");
        }
        assert_eq!(v.count(&p), 3);
    }

    #[test]
    fn strangers_other_tasks_repeats_and_the_model_itself_are_skipped() {
        let p = Pipeline::default();
        let v = Vary {
            models: vec![
                "fal/flux-schnell".into(),       // itself
                "fal/flux-kontext-pro".into(),   // an edit model: other ports
                "fal/from-the-future".into(),    // unknown here
                "fal/flux-dev".into(),
                "fal/flux-dev".into(),           // twice
            ],
            ..Default::default()
        };
        assert_eq!(ids(&v.expand(&p)), ["fal/flux-schnell", "fal/flux-dev"]);
    }

    #[test]
    fn alternatives_share_the_task_and_licence_is_asked_of_what_would_run() {
        let p = Pipeline::for_model("fal/flux-kontext-pro").unwrap();
        let alts = alternatives(&p);
        assert!(!alts.is_empty() && alts.iter().all(|m| m.task == Task::Edit && m.id != "fal/flux-kontext-pro"));
        let p = Pipeline::default();
        assert!(Vary { models: vec!["fal/nano-banana-2".into()], ..Default::default() }.all_commercial(&p));
        // FLUX dev is research-only: fanning out to it tags the node.
        assert!(!Vary { models: vec!["fal/flux-dev".into()], ..Default::default() }.all_commercial(&p));
        // ...but only when it would really run on this pipeline.
        let edit = Pipeline::for_model("fal/flux-kontext-pro").unwrap();
        assert!(Vary { models: vec!["fal/flux-dev".into()], ..Default::default() }.all_commercial(&edit));
    }

    #[test]
    fn the_fan_out_step_is_the_one_feeding_output() {
        // prompt -> schnell -> topaz -> out: the upscaler is what fans out.
        let p = crate::templates::BUILTIN[0].pipeline();
        assert_eq!(fan_model(&p).map(|m| m.id), Some("fal/topaz-upscale"));
        let v = Vary { models: vec!["fal/seedvr2-upscale".into()], ..Default::default() };
        assert_eq!(ids(&v.expand(&p)), ["fal/topaz-upscale", "fal/seedvr2-upscale"]);
    }

    fn prompts(v: &[Pipeline]) -> Vec<String> {
        v.iter()
            .map(|p| match &p.steps[prompt_step(p).unwrap()].op {
                Op::Prompt { text } => text.clone(),
                _ => "<wired>".into(),
            })
            .collect()
    }

    fn seeds(v: &[Pipeline]) -> Vec<u32> {
        v.iter()
            .map(|p| match &p.steps[fan_step(p).unwrap()].op {
                Op::Generate { seed, .. } => *seed,
                _ => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn prompts_seeds_and_models_multiply_grouped_for_reading() {
        let p = Pipeline::default(); // typed prompt "", seed 1
        let v = Vary {
            prompts: vec!["a moth".into(), "  ".into(), "a lamp".into()],
            seeds: 1,
            models: vec!["fal/flux-dev".into()],
            ..Default::default()
        };
        let all = v.expand(&p);
        assert_eq!(all.len(), 8);
        assert_eq!(v.count(&p), 8);
        assert_eq!(prompts(&all), ["a moth", "a moth", "a moth", "a moth", "a lamp", "a lamp", "a lamp", "a lamp"]);
        let s2 = next_seed(1);
        assert_eq!(seeds(&all), [1, 1, s2, s2, 1, 1, s2, s2]);
        assert_eq!(ids(&all)[..4], ["fal/flux-schnell", "fal/flux-dev", "fal/flux-schnell", "fal/flux-dev"]);
    }

    #[test]
    fn a_prompt_list_replaces_a_wired_prompt_and_is_ignored_without_one() {
        let mut wired = Pipeline::default();
        wired.steps[0].op = Op::PromptIn;
        let v = Vary { prompts: vec!["a moth".into()], ..Default::default() };
        assert_eq!(prompts(&v.expand(&wired)), ["a moth"]);
        // An upscaler has no prompt: the list makes no extra runs.
        let up = Pipeline::for_model("fal/esrgan").unwrap();
        assert_eq!(v.expand(&up), vec![up.clone()]);
        assert_eq!(v.count(&up), 1);
    }

    #[test]
    fn more_seeds_walk_the_panel_s_new_seed_sequence() {
        let p = Pipeline::default();
        let v = Vary { seeds: 2, ..Default::default() };
        assert_eq!(seeds(&v.expand(&p)), [1, next_seed(1), next_seed(next_seed(1))]);
    }

    #[test]
    fn frame_lists_read_numbers_ranges_and_steps() {
        assert_eq!(parse_frames("").unwrap(), Vec::<i64>::new());
        assert_eq!(parse_frames("1001, 1005 1001").unwrap(), [1001, 1005]);
        assert_eq!(parse_frames("10-13").unwrap(), [10, 11, 12, 13]);
        assert_eq!(parse_frames("1010-1050x20, 7").unwrap(), [1010, 1030, 1050, 7]);
        assert_eq!(parse_frames("-5--3").unwrap(), [-5, -4, -3]);
        assert!(parse_frames("ten").is_err());
        assert!(parse_frames("20-10").unwrap_err().contains("backwards"));
        assert!(parse_frames("1-9x0").is_err());
        assert!(parse_frames("1-1000").unwrap_err().contains("more than"));
        let v = Vary { frames: "1-3".into(), ..Default::default() };
        assert!(!v.is_empty());
        assert_eq!(v.ticks().unwrap(), [1, 2, 3]);
    }

    #[test]
    fn an_empty_vary_is_not_saved_and_old_setups_load() {
        assert_eq!(serde_json::to_string(&Vary::default()).unwrap(), "{}");
        assert_eq!(serde_json::from_str::<Vary>("{}").unwrap(), Vary::default());
    }
}
