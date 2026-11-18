//! App mode for a GenAI Pipeline (`genai::app`): the pipeline as a form.
//!
//! Two faces. While BUILDING, `author` is an "app…" menu under the step
//! list: tick the fields worth showing, name them, give the form a title,
//! switch app mode on. While USING, `form` is the node's whole left
//! column: the title, those fields, and nothing else — no step list, no
//! templates, and Enter does not open the pipeline (`Gfx::dive_into` asks
//! `AppForm::on`). `edit` on the form switches app mode off again; it is a
//! lock against accidents, not against people.
//!
//! The run column is the same one either way: the quote, pay & run and the
//! history do not change because the panel got simpler.

use egui::Ui;

use super::small;
use engine::graph::GenAiParams;
use genai::app::{self, What};

/// The id of the author menu's button, and of the form's `edit` button.
pub fn button_id(node: u64, what: &str) -> egui::Id {
    egui::Id::new(("genai-app", node, what))
}

/// The "app…" menu: choose the form's fields and switch app mode on.
pub(super) fn author(ui: &mut Ui, node: u64, p: &mut GenAiParams) {
    let r = ui.menu_button(small("app…"), |ui| {
        ui.set_max_width(280.0);
        p.app.prune(&p.pipeline);
        ui.weak(small("the fields the form shows; everything else is locked"));
        ui.add(
            egui::TextEdit::singleline(&mut p.app.title)
                .id(egui::Id::new(("genai-app-title", node)))
                .hint_text("title")
                .desired_width(f32::INFINITY),
        );
        let cands = app::candidates(&p.pipeline);
        if cands.is_empty() {
            ui.weak(small("nothing to show yet: the pipeline needs an Output its steps reach"));
        }
        for (i, what) in cands {
            let mut on = p.app.shows(i, what);
            let default = app::default_label(&p.pipeline, i, what);
            ui.horizontal(|ui| {
                if ui.checkbox(&mut on, "").changed() {
                    p.app.toggle(i, what, on);
                }
                match p.app.fields.iter_mut().find(|f| f.step == i && f.what == what) {
                    Some(f) => {
                        ui.add(
                            egui::TextEdit::singleline(&mut f.label)
                                .id(egui::Id::new(("genai-app-label", node, i, what as u8)))
                                .hint_text(default)
                                .desired_width(f32::INFINITY),
                        );
                    }
                    None => {
                        ui.weak(small(default));
                    }
                }
            });
        }
        ui.separator();
        ui.horizontal(|ui| {
            if ui.small_button("suggest").on_hover_text("every prompt, and the last model's seed").clicked() {
                let title = std::mem::take(&mut p.app.title);
                p.app = genai::AppForm { title, ..genai::AppForm::suggested(&p.pipeline) };
            }
            let ready = !p.app.fields.is_empty();
            let b = ui
                .add_enabled(ready, egui::Button::new(small("app mode")))
                .on_hover_text("show the form in place of the steps, and lock the pipeline")
                .on_disabled_hover_text("tick a field first");
            if b.clicked() {
                p.app.on = true;
                ui.close();
            }
        });
    });
    crate::widgets::note_rect(ui, button_id(node, "author"), r.response.rect);
    r.response.on_hover_text("turn this pipeline into a simple form with its other settings locked");
}

/// The form: the node's left column while app mode is on.
pub(super) fn form(ui: &mut Ui, node: u64, p: &mut GenAiParams) {
    p.app.prune(&p.pipeline);
    ui.horizontal(|ui| {
        if !p.app.title.is_empty() {
            ui.label(egui::RichText::new(&p.app.title).strong());
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let r = ui.small_button("edit").on_hover_text("leave app mode: the steps and every setting come back");
            crate::widgets::note_rect(ui, button_id(node, "edit"), r.rect);
            if r.clicked() {
                p.app.on = false;
            }
        });
    });
    let fields = p.app.fields.clone();
    for f in &fields {
        let label = if f.label.trim().is_empty() {
            app::default_label(&p.pipeline, f.step, f.what)
        } else {
            f.label.trim().to_string()
        };
        match f.what {
            What::Text => {
                ui.label(small(&label));
                if let Some(text) = app::text_mut(&mut p.pipeline, f.step) {
                    ui.add(
                        egui::TextEdit::multiline(text)
                            .id(egui::Id::new(("genai-app-text", node, f.step)))
                            .desired_rows(3)
                            .desired_width(f32::INFINITY),
                    );
                }
            }
            What::Seed => {
                let Some(mut v) = app::number(&p.pipeline, f.step, f.what) else { continue };
                ui.horizontal(|ui| {
                    ui.label(small(&label));
                    if ui.add(egui::DragValue::new(&mut v).update_while_editing(false).speed(1.0)).changed() {
                        app::set_number(&mut p.pipeline, f.step, f.what, v);
                    }
                    if ui.small_button("new").on_hover_text(crate::genaiui::priced!("a variation: a new request, billed again", "a variation: a new request")).clicked() {
                        let next = genai::vary::next_seed(v.max(0.0) as u32);
                        app::set_number(&mut p.pipeline, f.step, f.what, next as f32);
                    }
                });
            }
            _ => {
                let (Some(mut v), Some(r)) =
                    (app::number(&p.pipeline, f.step, f.what), app::range(&p.pipeline, f.step, f.what))
                else {
                    continue;
                };
                let before = v;
                crate::widgets::plain_slider(ui, &label, &mut v, r.lo, r.hi, false);
                if v != before {
                    app::set_number(&mut p.pipeline, f.step, f.what, v);
                }
            }
        }
    }
    if fields.is_empty() {
        ui.weak(small("this form has no fields left: press edit"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The form must fit the tiles the node asks for, like the full panel:
    /// a title, two prompts, a seed and a slider.
    #[test]
    fn the_form_fits_the_left_column() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(super::super::TILES) / 2.0 - 10.0, span(super::super::TILE_ROWS) - 36.0);
        let mut p = GenAiParams::new(0);
        p.pipeline = genai::templates::BUILTIN.iter().find(|t| t.name == "Describe and reimagine").unwrap().pipeline();
        p.app = genai::AppForm::suggested(&p.pipeline);
        p.app.title = "Reimagine".into();
        let g = genai::vary::fan_step(&p.pipeline).unwrap();
        p.app.toggle(g, What::Megapixels, true);
        p.app.on = true;
        assert!(p.app.fields.len() >= 4, "{:?}", p.app.fields);
        let over = std::cell::Cell::new(0.0f32);
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| form(ui, 1, &mut p));
            over.set(out.content_size.y - out.inner_rect.height());
        });
        shot.save("genai-app-form");
        assert!(over.get() <= 0.5, "the form overflows its column by {}pt", over.get());
    }
}
