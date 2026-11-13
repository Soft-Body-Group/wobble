//! `--test-plugin`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Tier 0 plugins end to end: the in-repo `plugins/invert` manifest is
    /// already loaded at startup; a Custom node joins the live chain, its
    /// shader compiles (fused, naga-gated), its param animates through the
    /// standard channel machinery, and an UNINSTALLED type still renders as
    /// a passthrough.
    pub(crate) fn drive_test_plugin(&mut self) {
        let Some((stage, mut frames, node)) = self.test_plugin else { return };
        frames += 1;
        if frames > 2400 {
            println!("PLUGIN TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                if engine::plugins::get("invert").is_none() {
                    println!(
                        "PLUGIN TEST FAIL: plugins/invert not loaded from the repo dir"
                    );
                    std::process::exit(1);
                }
                // Splice invert after the grade and view it.
                let id = self.egraph.add(
                    NodeKind::Custom { type_id: "invert".into(), params: Default::default() },
                    egui::pos2(320.0, 40.0),
                );
                self.egraph.connect(id, 0, 3);
                self.egraph.view = Some(id);
                self.test_plugin = Some((1, frames, id));
            }
            // The fused pipeline with the plugin in it must come up.
            1 if frames > 20 => {
                if !self.last_pipeline_ready {
                    self.test_plugin = Some((1, frames, node));
                    return;
                }
                // Key the param: frame 0 -> 0, frame 10 -> 1. The evaluated
                // graph must follow the curve while authored stays put.
                let ch = anim::Channel::Custom(anim::intern("amount"));
                if let Some(n) = self.egraph.node_mut(node) {
                    let c = n.anim.entry(ch).or_default();
                    c.set_key(0, 0.0);
                    c.set_key(10, 1.0);
                }
                let at5 = self
                    .egraph
                    .to_engine_graph_at(Some(5))
                    .nodes
                    .get(&node)
                    .and_then(|k| anim::get(k, ch));
                let authored = self
                    .egraph
                    .node(node)
                    .and_then(|n| anim::get(&n.kind, ch));
                if at5 != Some(0.5) {
                    println!("PLUGIN TEST FAIL: keyed param evals {at5:?}, want 0.5");
                    std::process::exit(1);
                }
                if authored != Some(1.0) {
                    println!(
                        "PLUGIN TEST FAIL: authored default {authored:?}, want 1.0 (untouched)"
                    );
                    std::process::exit(1);
                }
                // Now a node whose plugin does not exist: must still render.
                let ghost = self.egraph.add(
                    NodeKind::Custom {
                        type_id: "not_installed".into(),
                        params: Default::default(),
                    },
                    egui::pos2(320.0, 120.0),
                );
                self.egraph.connect(ghost, 0, node);
                self.egraph.view = Some(ghost);
                self.test_plugin = Some((2, frames, node));
            }
            2 if frames > 12 => {
                if !self.last_pipeline_ready {
                    self.test_plugin = Some((2, frames, node));
                    return;
                }
                println!(
                    "PLUGIN TEST PASS: invert loaded from plugins/, compiled into the fused pass, param keyed 0->1 evals 0.5 at frame 5 (authored untouched); uninstalled type renders as passthrough"
                );
                std::process::exit(0);
            }
            _ => self.test_plugin = Some((stage, frames, node)),
        }
    }
}
