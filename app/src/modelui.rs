//! The Models panel and the per-node model row.
//!
//! Weights are the one thing in Wobble a user has to acquire rather than
//! build, so acquiring them is a first-class bit of UI rather than a README
//! step: what it is, who wrote it, what licence it carries, how big it is,
//! and one button. Three views of one cache, all driven by what nodes
//! declare in `NodeKind::models()` — no ML node draws any of this itself:
//!
//! - `node_gate` — top of the node's properties panel: the missing weights,
//!   one button for all of them, so a node that cannot run says exactly why
//!   and fixes itself in place;
//! - `setup_prompt` — a comp opened with weights missing lists them once;
//! - `panel_ui` — the whole cache in a window (the menubar's "models…").
//!
//! Nothing here starts a download on its own (see `engine::models`) — every
//! fetch is a click, and the size is on the button before you press it.

use crate::theme;
use engine::models::{self, Download, ModelSpec, State};

/// Downloads in flight, keyed by model id. One per model: clicking a second
/// time while one runs is a cancel, not a second connection.
#[derive(Default)]
pub struct Downloads {
    running: Vec<(&'static str, Download)>,
    /// Last failure per model, kept until the next attempt so the message
    /// survives the frame the job finished on.
    errors: Vec<(&'static str, String)>,
}

impl Downloads {
    /// Drain every running job. Returns true while any is still going, so
    /// the caller knows to keep the UI repainting.
    pub fn poll(&mut self) -> bool {
        for (id, dl) in &mut self.running {
            if !dl.poll() {
                if let Some(e) = dl.error.clone() {
                    self.errors.retain(|(k, _)| k != id);
                    self.errors.push((id, e));
                }
            }
        }
        self.running.retain(|(_, dl)| !dl.finished);
        !self.running.is_empty()
    }

    pub fn busy(&self) -> bool {
        !self.running.is_empty()
    }

    fn job(&self, id: &str) -> Option<&Download> {
        self.running.iter().find(|(k, _)| *k == id).map(|(_, d)| d)
    }

    /// Fetch every one of these that is not already on its way.
    pub fn start_all(&mut self, specs: &[&'static ModelSpec]) {
        for s in specs {
            self.start(s);
        }
    }

    fn start(&mut self, spec: &'static ModelSpec) {
        if self.job(spec.id).is_some() {
            return;
        }
        self.errors.retain(|(k, _)| *k != spec.id);
        self.running.push((spec.id, models::fetch(spec)));
    }

    fn stop(&mut self, id: &str) {
        // Dropping the job cancels it; the .part file stays for a resume.
        self.running.retain(|(k, dl)| {
            if *k == id {
                dl.cancel();
                false
            } else {
                true
            }
        });
    }

    fn error(&self, id: &str) -> Option<&str> {
        self.errors.iter().find(|(k, _)| *k == id).map(|(_, e)| e.as_str())
    }
}

/// One model: what it is, and the button that gets it. Returns true if this
/// frame changed what is on disk (so a node can re-check and clear a stale
/// "no weights" state).
pub fn model_row(ui: &mut egui::Ui, spec: &'static ModelSpec, dl: &mut Downloads) -> bool {
    let mut landed = false;
    let state = models::state(spec);
    let running = dl.job(spec.id).is_some();

    ui.horizontal(|ui| {
        ui.strong(spec.label);
        ui.weak(spec.size_label());
        // The status word carries the whole meaning, so it is coloured
        // rather than left to be read.
        match &state {
            State::Ready if running => {}
            State::Ready => {
                ui.colored_label(theme::ok(), "cached");
            }
            State::Partial { have } => {
                if !running {
                    ui.colored_label(
                        theme::warn(),
                        format!("part {}", models::human_bytes(*have)),
                    );
                }
            }
            State::Bad(_) => {
                ui.colored_label(theme::error(), "wrong size");
            }
            State::Missing => {}
        }
    });
    ui.weak(spec.about);

    if running {
        let job = dl.job(spec.id).expect("checked above");
        let frac = job.fraction().unwrap_or(0.0);
        let text = format!(
            "{} / {}",
            models::human_bytes(job.have),
            models::human_bytes(job.total)
        );
        ui.horizontal(|ui| {
            ui.add(egui::ProgressBar::new(frac).desired_width(220.0).text(text));
            if ui.button("cancel").on_hover_text("keeps the part file — a later fetch resumes").clicked() {
                dl.stop(spec.id);
            }
        });
    } else {
        ui.horizontal(|ui| {
            let label = match &state {
                State::Ready => "re-download".to_string(),
                State::Partial { have } => format!(
                    "resume ({} to go)",
                    models::human_bytes(spec.bytes.saturating_sub(*have))
                ),
                _ => format!("download {}", spec.size_label()),
            };
            if ui
                .button(label)
                .on_hover_text(format!("{}\n{} · {}", spec.url, spec.license, spec.source))
                .clicked()
            {
                if state == State::Ready {
                    // An explicit re-download replaces the file; clear it
                    // first so a failure cannot leave the old one looking
                    // like the new one.
                    let _ = std::fs::remove_file(spec.path());
                }
                dl.start(spec);
                landed = true;
            }
            if matches!(state, State::Ready | State::Bad(_) | State::Partial { .. })
                && ui.button("remove").clicked()
            {
                let _ = std::fs::remove_file(spec.path());
                let _ = std::fs::remove_file(spec.part_path());
                landed = true;
            }
        });
    }

    ui.weak(format!("{} · {}", spec.license, spec.source));
    if let State::Bad(why) = &state {
        ui.colored_label(theme::error(), why);
    }
    if let Some(e) = dl.error(spec.id) {
        ui.colored_label(theme::error(), e);
    }
    if !spec.verifiable() {
        ui.weak("no checksum pinned for this release — size is all that is verified");
    }
    landed
}

/// The whole cache in one window: every model, grouped by the node it feeds.
pub fn panel_ui(ui: &mut egui::Ui, dl: &mut Downloads) {
    ui.horizontal(|ui| {
        ui.weak("cache");
        let dir = models::models_dir();
        ui.monospace(dir.display().to_string());
        if ui.small_button("reveal").on_hover_text("open in Finder").clicked() {
            let _ = std::fs::create_dir_all(&dir);
            // Best effort: a missing opener is not worth an error dialog.
            #[cfg(target_os = "macos")]
            let _ = std::process::Command::new("open").arg(&dir).spawn();
        }
    });
    ui.weak("override with $WOBBLE_MODELS_DIR");
    ui.separator();

    for family in models::families() {
        ui.label(egui::RichText::new(family).weak());
        for spec in models::all().iter().filter(|m| m.family == family) {
            model_row(ui, spec, dl);
            ui.separator();
        }
    }
    ui.weak(
        "weights are downloaded from their upstream publishers and cached per user; \
         nothing is fetched until you ask for it",
    );
}

/// The download gate a node wears at the top of its properties panel,
/// drawn for any node whose `NodeKind::models()` is non-empty. Returns true
/// when every weight is ready — the node's own panel follows either way (a
/// node is set up before its weights land), but nothing it does will run.
pub fn node_gate(ui: &mut egui::Ui, ids: &[&str], dl: &mut Downloads) -> bool {
    let missing = models::missing(ids.iter().copied());
    if missing.is_empty() {
        let names: Vec<&str> = models::specs(ids.iter().copied()).iter().map(|m| m.label).collect();
        ui.horizontal(|ui| {
            ui.colored_label(theme::ok(), "weights ready");
            ui.weak(names.join(", "));
        });
        return true;
    }
    egui::Frame::group(ui.style()).show(ui, |ui| {
        ui.colored_label(
            theme::warn(),
            "needs downloaded weights — this node does nothing until they land",
        );
        let waiting: Vec<&'static ModelSpec> =
            missing.iter().copied().filter(|m| dl.job(m.id).is_none()).collect();
        if waiting.len() > 1 {
            let bytes = models::total_bytes(&waiting);
            if ui.button(format!("download all ({})", models::human_bytes(bytes))).clicked() {
                dl.start_all(&waiting);
            }
        }
        for m in &missing {
            ui.separator();
            model_row(ui, m, dl);
        }
    });
    false
}

/// Weights a freshly opened setup needs but the cache lacks, if any — what
/// `setup_prompt` shows. Disabled nodes count: switching one back on should
/// not be the moment you find out.
pub fn needed_by<'a>(kinds: impl IntoIterator<Item = &'a engine::graph::NodeKind>) -> Vec<&'static ModelSpec> {
    models::missing(kinds.into_iter().flat_map(|k| k.models().iter().copied()))
}

/// "This setup needs N models": shown once, after a load that came up
/// short. Closes itself when everything has landed, or on "later" — the
/// nodes' own gates still say what is missing.
pub fn setup_prompt(ctx: &egui::Context, needed: &mut Vec<&'static ModelSpec>, dl: &mut Downloads) {
    if needed.is_empty() {
        return;
    }
    needed.retain(|m| models::state(m) != State::Ready);
    if needed.is_empty() {
        return;
    }
    let mut open = true;
    let mut later = false;
    egui::Window::new("missing weights")
        .open(&mut open)
        .collapsible(false)
        .default_width(420.0)
        .show(ctx, |ui| {
            ui.label(format!(
                "this setup uses {} model{} not in the cache ({})",
                needed.len(),
                if needed.len() == 1 { "" } else { "s" },
                models::human_bytes(models::total_bytes(needed)),
            ));
            ui.horizontal(|ui| {
                if ui.button("download all").clicked() {
                    dl.start_all(needed);
                }
                if ui.button("later").clicked() {
                    later = true;
                }
            });
            for m in needed.iter() {
                ui.separator();
                model_row(ui, m, dl);
            }
        });
    if !open || later {
        needed.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The panel must draw with an empty cache and with a job in flight —
    /// the two states a first-run user actually sees.
    #[test]
    fn panel_draws_headless() {
        let ctx = egui::Context::default();
        let mut dl = Downloads::default();
        for _ in 0..2 {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| panel_ui(ui, &mut dl));
        }
        assert!(!dl.busy(), "the panel must not start a download by itself");
    }

    /// The gate and the prompt draw with weights missing and must not
    /// fetch anything by themselves; an empty declaration draws nothing
    /// to fetch and reports ready. Points WOBBLE_MODELS_DIR at an empty dir
    /// so a developer's real cache cannot make "missing" pass as "ready".
    #[test]
    fn gate_and_prompt_draw_headless_without_fetching() {
        let dir = std::env::temp_dir().join(format!("wobble-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("WOBBLE_MODELS_DIR", &dir);
        let ctx = egui::Context::default();
        let mut dl = Downloads::default();
        let ids = ["bootstapir-causal", "sam2-tiny-decoder"];
        let mut ready = true;
        let mut empty_ready = false;
        let mut needed = models::missing(ids);
        assert_eq!(needed.len(), 2);
        for _ in 0..2 {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                ready = node_gate(ui, &ids, &mut dl);
                empty_ready = node_gate(ui, &[], &mut dl);
            });
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| setup_prompt(ui.ctx(), &mut needed, &mut dl));
        }
        std::env::remove_var("WOBBLE_MODELS_DIR");
        assert!(!ready, "missing weights must not read as ready");
        assert!(empty_ready, "a node declaring nothing is never gated");
        assert_eq!(needed.len(), 2, "the prompt stays until answered");
        assert!(!dl.busy(), "nothing downloads without a click");
    }
}
