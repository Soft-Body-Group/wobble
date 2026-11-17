//! Variations: one run of a node that makes SEVERAL takes.
//!
//! A [`Vary`] sits beside the pipeline and says how to multiply it. Today
//! it holds one axis:
//!
//! - **models** (fan-out): the same prompt, seed and inputs through other
//!   models as well, so their results land side by side in the history.
//!
//! [`Vary::expand`] turns a pipeline into its variants, the pipeline itself
//! first. Each variant is an ordinary pipeline: planned, quoted, cached and
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
        self.models.is_empty()
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

    /// Every pipeline a run walks: `p` itself, then one per fan-out model.
    pub fn expand(&self, p: &Pipeline) -> Vec<Pipeline> {
        let mut out = vec![p.clone()];
        let Some(step) = fan_step(p) else { return out };
        for m in self.models_for(p) {
            let mut v = p.clone();
            v.set_model(step, m.id);
            out.push(v);
        }
        out
    }

    /// How many takes a run of `p` makes.
    pub fn count(&self, p: &Pipeline) -> usize {
        1 + self.models_for(p).len()
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
        let v = Vary { models: vec!["fal/flux-dev".into(), "fal/nano-banana-2".into()] };
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
        };
        assert_eq!(ids(&v.expand(&p)), ["fal/flux-schnell", "fal/flux-dev"]);
    }

    #[test]
    fn alternatives_share_the_task_and_licence_is_asked_of_what_would_run() {
        let p = Pipeline::for_model("fal/flux-kontext-pro").unwrap();
        let alts = alternatives(&p);
        assert!(!alts.is_empty() && alts.iter().all(|m| m.task == Task::Edit && m.id != "fal/flux-kontext-pro"));
        let p = Pipeline::default();
        assert!(Vary { models: vec!["fal/nano-banana-2".into()] }.all_commercial(&p));
        // FLUX dev is research-only: fanning out to it tags the node.
        assert!(!Vary { models: vec!["fal/flux-dev".into()] }.all_commercial(&p));
        // ...but only when it would really run on this pipeline.
        let edit = Pipeline::for_model("fal/flux-kontext-pro").unwrap();
        assert!(Vary { models: vec!["fal/flux-dev".into()] }.all_commercial(&edit));
    }

    #[test]
    fn the_fan_out_step_is_the_one_feeding_output() {
        // prompt -> schnell -> topaz -> out: the upscaler is what fans out.
        let p = crate::templates::BUILTIN[0].pipeline();
        assert_eq!(fan_model(&p).map(|m| m.id), Some("fal/topaz-upscale"));
        let v = Vary { models: vec!["fal/seedvr2-upscale".into()] };
        assert_eq!(ids(&v.expand(&p)), ["fal/topaz-upscale", "fal/seedvr2-upscale"]);
    }

    #[test]
    fn an_empty_vary_is_not_saved_and_old_setups_load() {
        assert_eq!(serde_json::to_string(&Vary::default()).unwrap(), "{}");
        assert_eq!(serde_json::from_str::<Vary>("{}").unwrap(), Vary::default());
    }
}
