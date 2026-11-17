//! The GenAI Pipeline's TEMPLATE shelf: whole pipelines, picked from a
//! menu in the panel (`genai::templates`).
//!
//! Built-in ones, then yours (`~/.wobble/genai-templates`). Picking one
//! replaces the pipeline's steps and nothing else: the takes stay, nothing
//! runs, and it is one undo step. "save as" writes the pipeline as it
//! stands into the folder, where it is a small JSON file you can send to
//! someone; dropped into THEIR folder it is on their shelf.
//!
//! A template whose models this build cannot offer (video in a build that
//! cannot play one) is listed greyed, with the reason, never hidden: a
//! shelf that changes length between machines reads as a bug.

use egui::Ui;

use super::{offered, small};
use engine::graph::GenAiParams;
use genai::templates;

/// Can this build run every model these steps name?
fn runnable(steps: &[genai::Step]) -> bool {
    templates::models_of(steps).into_iter().all(offered)
}

/// The id of the shelf's menu button, for a suite to aim at.
pub fn button_id(node: u64) -> egui::Id {
    egui::Id::new(("genai-templates", node))
}

/// The shelf, as a menu button under the pipeline's step list.
pub(super) fn menu(ui: &mut Ui, node: u64, p: &mut GenAiParams) {
    let name_id = egui::Id::new(("genai-template-name", node));
    let note_id = egui::Id::new(("genai-template-note", node));
    let r = ui.menu_button(small("templates…"), |ui| {
        ui.set_max_width(260.0);
        for t in templates::BUILTIN {
            let pl = t.pipeline();
            let ok = runnable(&pl.steps);
            let r = ui.add_enabled(ok, egui::Button::new(small(t.name)));
            let r = r.on_hover_text(t.about).on_disabled_hover_text(format!(
                "{}\n(this build cannot run one of its models: video needs --features app/movie)",
                t.about
            ));
            if r.clicked() {
                p.pipeline.steps = pl.steps;
                ui.close();
            }
        }
        ui.separator();
        let dir = templates::user_dir();
        let mine = templates::list(&dir);
        if mine.is_empty() {
            ui.weak(small("none of your own yet"));
        }
        for (path, t) in mine {
            let about = if t.about.is_empty() { path.to_string_lossy().into_owned() } else { t.about.clone() };
            let r = ui.add_enabled(runnable(&t.steps), egui::Button::new(small(&t.name)));
            if r.on_hover_text(about).clicked() {
                p.pipeline.steps = t.steps;
                ui.close();
            }
        }
        ui.separator();
        let mut name: String = ui.data(|d| d.get_temp(name_id)).unwrap_or_default();
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut name).hint_text("name").desired_width(130.0));
            if ui.add_enabled(!name.trim().is_empty(), egui::Button::new(small("save as"))).clicked() {
                let note = match templates::save(&dir, &name, "", &p.pipeline.steps) {
                    Ok(f) => format!("saved {}", f.display()),
                    Err(e) => format!("could not save: {e}"),
                };
                ui.data_mut(|d| d.insert_temp(note_id, note));
                name.clear();
            }
        });
        ui.data_mut(|d| d.insert_temp(name_id, name));
        if let Some(note) = ui.data(|d| d.get_temp::<String>(note_id)) {
            ui.weak(small(note));
        }
        ui.weak(small("a template is a file in ~/.wobble/genai-templates: send it to share it"));
    });
    crate::widgets::note_rect(ui, button_id(node), r.response.rect);
    r.response.on_hover_text("replace the steps with a ready-made pipeline, or save this one");
}

/// The built-in templates this build can run, for the add menu: each makes
/// a GenAI Pipeline node already wired that way.
pub fn offered_templates() -> impl Iterator<Item = &'static templates::Template> {
    templates::BUILTIN.iter().filter(|t| runnable(&t.pipeline().steps))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_add_menu_offers_only_templates_this_build_can_run() {
        for t in offered_templates() {
            assert!(t.models().into_iter().all(offered), "{}", t.name);
        }
        // A still-only template runs in every build.
        assert!(offered_templates().any(|t| t.name == "Clean plate"));
    }
}
