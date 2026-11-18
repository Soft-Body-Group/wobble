//! The fan-out row of a GenAI node's panel ("also run on…") and, under
//! it, the batch section: a list of prompts, more seeds, a list of frames.
//! They multiply each other, and the line beneath says how many takes a
//! run will make before `run…` is pressed.
//!
//! One prompt, one seed, one set of inputs, through several models: tick
//! the others here and `run…` plans, quotes and runs them all (one price,
//! one confirm), each landing its own take. The history is where they sit
//! side by side; its `sheet` shows them large.
//!
//! Only models of the SAME task as the pipeline's fan-out step are offered
//! (`genai::vary::alternatives`): the same ports, and a result that means
//! the same thing. A research-only model is offered in the licence colour
//! and tags the node the moment it is ticked, before anything is sent.

use egui::Ui;

use super::{offered, small};
use crate::theme;
use engine::graph::GenAiParams;
use genai::vary;

/// The id of the row's menu button, for a suite to aim at.
pub fn button_id(node: u64) -> egui::Id {
    egui::Id::new(("genai-fanout", node))
}

/// The id of the batch section's header (global, so a test can open it).
pub fn batch_id(node: u64) -> egui::Id {
    egui::Id::new(("genai-batch", node))
}

/// Does a run of this pipeline render any of the node's inputs? Only then
/// does a frame list multiply it.
fn captures(pl: &genai::Pipeline) -> bool {
    !pl.reads().is_empty()
}

/// How many takes a run would make, or why it cannot be counted.
pub fn runs(p: &GenAiParams) -> Result<usize, String> {
    let frames = if captures(&p.pipeline) { p.vary.ticks()?.len().max(1) } else { 1 };
    Ok(p.vary.count(&p.pipeline) * frames)
}

pub(super) fn row(ui: &mut Ui, node: u64, p: &mut GenAiParams) {
    fan_out(ui, node, p);
    batch(ui, node, p);
    match runs(p) {
        Ok(n) if n > vary::MAX_RUNS => {
            ui.colored_label(
                theme::warn(),
                small(format!("{n} runs is more than {} in one go: shorten a list", vary::MAX_RUNS)),
            );
        }
        Ok(n) if n > 1 => {
            ui.weak(small(crate::genaiui::priced!(
                format!("run… makes {n} takes, priced together"),
                format!("run… makes {n} takes")
            )));
        }
        Ok(_) => {}
        Err(e) => {
            ui.colored_label(theme::error(), small(format!("frames: {e}")));
        }
    }
}

/// The batch section: a prompt list, more seeds, a frame list. Shut by
/// default; its header says when something inside is set.
fn batch(ui: &mut Ui, node: u64, p: &mut GenAiParams) {
    let v = &mut p.vary;
    let set = !v.prompts.is_empty() || v.seeds > 0 || !v.frames.trim().is_empty();
    let st = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), batch_id(node), false);
    st.show_header(ui, |ui| {
        ui.label(small(if set { "batch… (on)" } else { "batch…" }))
            .on_hover_text("run the node over a list of prompts, more seeds, or a list of frames");
    })
    .body(|ui| {
        let has_prompt = vary::prompt_step(&p.pipeline).is_some();
        let mut text = v.prompts.join("\n");
        let r = ui.add_enabled(
            has_prompt,
            egui::TextEdit::multiline(&mut text)
                .id(egui::Id::new(("genai-batch-prompts", node)))
                .desired_rows(2)
                .desired_width(f32::INFINITY)
                .hint_text("prompts, one per line: each runs in place of the node's own"),
        );
        if r.on_disabled_hover_text("this pipeline has no prompt").changed() {
            v.prompts = if text.is_empty() { vec![] } else { text.split('\n').map(String::from).collect() };
        }
        ui.horizontal(|ui| {
            ui.label(small("more seeds"));
            let mut n = v.seeds as i32;
            if ui.add(egui::DragValue::new(&mut n).range(0..=16).speed(0.05)).changed() {
                v.seeds = n.clamp(0, 16) as u32;
            }
        });
        ui.horizontal(|ui| {
            ui.label(small("frames"))
                .on_hover_text("capture the inputs at each of these frames instead of the playhead: a Load of stills is a list of images");
            ui.add(
                egui::TextEdit::singleline(&mut v.frames)
                    .id(egui::Id::new(("genai-batch-frames", node)))
                    .hint_text("1001, 1010-1050x10")
                    .desired_width(f32::INFINITY),
            );
        });
        if set && ui.small_button("clear").clicked() {
            v.prompts.clear();
            v.seeds = 0;
            v.frames.clear();
        }
    });
}

fn fan_out(ui: &mut Ui, node: u64, p: &mut GenAiParams) {
    let alts: Vec<_> = vary::alternatives(&p.pipeline).into_iter().filter(|m| offered(m)).collect();
    if alts.is_empty() {
        // Nothing to fan out to (a one-of-a-kind model, or no model step):
        // no row, rather than a dead button.
        return;
    }
    let n = p.vary.models_for(&p.pipeline).len();
    let own = vary::fan_model(&p.pipeline).map(|m| m.short).unwrap_or("model");
    let label = if n == 0 { "also run on…".to_string() } else { format!("{own} + {n} more…") };
    let r = ui.menu_button(small(label), |ui| {
        ui.set_max_width(240.0);
        ui.weak(small("the same prompt, seed and inputs, through each ticked model"));
        for m in &alts {
            let mut on = p.vary.models.iter().any(|id| id == m.id);
            let mut text = small(m.label);
            if !m.commercial {
                text = text.color(theme::cost_nc());
            }
            let hover = format!("{}\n{}", m.about, m.licence);
            if ui.checkbox(&mut on, text).on_hover_text(hover).changed() {
                p.vary.models.retain(|id| id != m.id);
                if on {
                    p.vary.models.push(m.id.to_string());
                }
            }
        }
        if n > 0 && ui.small_button("none").clicked() {
            p.vary.models.clear();
        }
    });
    crate::widgets::note_rect(ui, button_id(node), r.response.rect);
    r.response.on_hover_text(crate::genaiui::priced!(
        "fan out: run… then makes one take per model, priced together",
        "fan out: run… then makes one take per model"
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_multiply_only_a_run_that_renders_an_input() {
        // Text to image: nothing is captured, so a frame list adds nothing.
        let mut t2i = GenAiParams::new(0);
        t2i.vary.frames = "1-4".into();
        t2i.vary.seeds = 1;
        assert_eq!(runs(&t2i), Ok(2));
        // An edit reads the plate: four frames, two seeds each.
        let mut edit = GenAiParams::for_model(0, "fal/flux-kontext-pro");
        edit.vary = t2i.vary.clone();
        assert_eq!(runs(&edit), Ok(8));
        edit.vary.frames = "four".into();
        assert!(runs(&edit).is_err());
    }
}
