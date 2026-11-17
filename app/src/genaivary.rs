//! The fan-out row of a GenAI node's panel: "also run on…".
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

pub(super) fn row(ui: &mut Ui, node: u64, p: &mut GenAiParams) {
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
    r.response.on_hover_text("fan out: run… then makes one take per model, priced together");
    if n > 0 {
        ui.weak(small(format!("run… makes {} takes", n + 1)));
    }
}
