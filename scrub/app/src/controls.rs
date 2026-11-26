//! The controls panel: the selected clip's (else the top clip under the
//! playhead's) speed, and the effects added to it from here — a
//! transform (fit / fill, scale, position, rotation, flips) and a colour
//! correction — with no trip into Wobble. Both are drawn by the record
//! viewer's display pass and the renderer's (one shader: `viewer.rs`), so
//! they cost a few ALU ops per pixel and never a decode. With the `fold`
//! feature, a wob clip's own published controls follow underneath.
//!
//! Drags are bursts: one undo step once they settle.

use crate::editor::Editor;
use crate::timelineui::Selection;
use egui::{Rect, RichText};
use model::textanim::{Ease, RangeShape, TextAnim, TextUnit};

/// What an effect row wants done when it is drawn.
#[derive(PartialEq)]
enum Head {
    Keep,
    Reset,
    Remove,
}

/// An effect's heading: its name, reset and remove buttons.
fn heading(ui: &mut egui::Ui, name: &str, on: Option<&mut bool>) -> Head {
    let mut h = Head::Keep;
    ui.horizontal(|ui| {
        match on {
            Some(on) => {
                ui.checkbox(on, RichText::new(name).strong()).on_hover_text("bypass it (keeps the settings)");
            }
            None => {
                ui.label(RichText::new(name).strong());
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("✕").on_hover_text("remove it").clicked() {
                h = Head::Remove;
            }
            if ui.small_button("reset").clicked() {
                h = Head::Reset;
            }
        });
    });
    h
}

/// A labelled slider row; true when it moved.
fn slider(ui: &mut egui::Ui, label: &str, v: &mut f64, range: std::ops::RangeInclusive<f64>, neutral: f64, suffix: &str) -> bool {
    let mut moved = false;
    ui.horizontal(|ui| {
        let r = ui.add_sized([74.0, 18.0], egui::Label::new(label).sense(egui::Sense::click())).on_hover_text("double-click: back to neutral");
        if r.double_clicked() && *v != neutral {
            *v = neutral;
            moved = true;
        }
        moved |= ui.add(egui::Slider::new(v, range).suffix(suffix).max_decimals(2).clamping(egui::SliderClamping::Never)).changed();
    });
    moved
}

impl Editor {
    pub(crate) fn controls_panel(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let used = ui
            .scope_builder(egui::UiBuilder::new().max_rect(rect), |ui| {
                ui.set_clip_rect(rect.intersect(ui.clip_rect()));
                egui::ScrollArea::vertical().id_salt("clip-controls").max_height(rect.height() * 0.75).auto_shrink([false, true]).show(ui, |ui| {
                    self.clip_controls(ui);
                });
            })
            .response
            .rect;
        #[cfg(feature = "fold")]
        {
            let wob = self.clip_target().is_some_and(|(t, i)| self.rec.seq.tracks[t].clips[i].wob.is_some());
            let rest = Rect::from_min_max(egui::pos2(rect.min.x, used.max.y + 8.0), rect.max);
            if wob && rest.height() > 40.0 {
                ui.painter().text(rest.min, egui::Align2::LEFT_TOP, "wob", egui::FontId::proportional(12.0), ui.visuals().strong_text_color());
                self.fold_controls(ui, Rect::from_min_max(egui::pos2(rest.min.x, rest.min.y + 18.0), rest.max));
            }
        }
        #[cfg(not(feature = "fold"))]
        let _ = used;
    }

    fn clip_controls(&mut self, ui: &mut egui::Ui) {
        let Some((t, i)) = self.clip_target() else {
            ui.label(RichText::new("select a clip (D, or click it) to see its controls").weak().small());
            return;
        };
        let c = self.rec.seq.tracks[t].clips[i].clone();
        let name = match &c.text {
            Some(x) => format!("text: {}", x.text.lines().next().unwrap_or("")),
            None => self.rec.seq.media.get(c.media).map_or(String::new(), |m| m.name.clone()),
        };
        ui.label(RichText::new(format!("{}  ·  {name}", self.rec.seq.tracks[t].name)).strong());
        let video = self.rec.seq.tracks[t].kind == model::Kind::Video && c.text.is_none();

        // Speed: what cmd+R / a stretch set, editable here.
        ui.horizontal(|ui| {
            ui.add_sized([74.0, 18.0], egui::Label::new("speed"));
            let id = ui.id().with(("clip-speed", c.link, c.start));
            let mut v: f64 = ui.data(|d| d.get_temp(id)).unwrap_or(c.speed);
            let r = ui.add(egui::DragValue::new(&mut v).speed(0.01).range(-16.0..=16.0).suffix("x").max_decimals(3)).on_hover_text("drag or type; the clip's length follows (cmd R)");
            let r_pct = ui.label(RichText::new(format!("{:.0}%", v * 100.0)).weak());
            let _ = r_pct;
            if c.conform {
                ui.label(RichText::new("conformed").weak().small());
            }
            if (r.drag_stopped() || r.lost_focus()) && v != c.speed {
                ui.data_mut(|d| d.remove::<f64>(id));
                if v != 0.0 {
                    self.apply_speed(v);
                }
            } else if r.dragged() || r.has_focus() {
                ui.data_mut(|d| d.insert_temp(id, v));
            } else {
                ui.data_mut(|d| d.remove::<f64>(id));
            }
        });
        if let Some(ct) = &c.text {
            self.text_anim_controls(ui, t, i, &ct.anims);
            return;
        }
        if !video {
            return;
        }

        // Which effects show: the ones with settings, and ones just added.
        let added = ui.id().with(("xf-added", c.link));
        let xf_on = !c.reframe.is_default() || ui.data(|d| d.get_temp::<bool>(added)).unwrap_or(false);
        ui.separator();
        ui.menu_button("+ add effect", |ui| {
            if ui.add_enabled(c.grade.is_none(), egui::Button::new("colour correction")).clicked() {
                self.set_grade_quiet(t, i, Some(model::Grade::default()), false);
                ui.close();
            }
            if ui.add_enabled(!xf_on, egui::Button::new("transform")).clicked() {
                ui.data_mut(|d| d.insert_temp(added, true));
                ui.close();
            }
        });

        if xf_on {
            ui.separator();
            let mut r = c.reframe;
            match heading(ui, "transform", None) {
                Head::Reset => r = model::Reframe::default(),
                Head::Remove => {
                    r = model::Reframe::default();
                    ui.data_mut(|d| d.remove::<bool>(added));
                }
                Head::Keep => {}
            }
            ui.horizontal(|ui| {
                ui.add_sized([74.0, 18.0], egui::Label::new("frame"));
                ui.selectable_value(&mut r.fill, false, "fit");
                ui.selectable_value(&mut r.fill, true, "fill");
                ui.checkbox(&mut r.flip_x, "flip H");
                ui.checkbox(&mut r.flip_y, "flip V");
            });
            let mut scale = r.zoom * 100.0;
            if slider(ui, "scale", &mut scale, 10.0..=400.0, 100.0, "%") {
                r.zoom = (scale / 100.0).clamp(0.05, 20.0);
            }
            let (mut x, mut y) = (r.x * 100.0, r.y * 100.0);
            if slider(ui, "position x", &mut x, -100.0..=100.0, 0.0, "%") {
                r.x = (x / 100.0).clamp(-2.0, 2.0);
            }
            if slider(ui, "position y", &mut y, -100.0..=100.0, 0.0, "%") {
                r.y = (y / 100.0).clamp(-2.0, 2.0);
            }
            slider(ui, "rotation", &mut r.rotate, -180.0..=180.0, 0.0, "°");
            if r != c.reframe {
                self.set_reframe_quiet(t, i, r);
            }
        }

        if let Some(g0) = c.grade {
            ui.separator();
            let mut g = g0;
            let mut on = !g.off;
            let head = heading(ui, "colour correction", Some(&mut on));
            g.off = !on;
            ui.add_enabled_ui(on, |ui| {
                slider(ui, "exposure", &mut g.exposure, -4.0..=4.0, 0.0, " st");
                slider(ui, "temperature", &mut g.temperature, -1.0..=1.0, 0.0, "");
                slider(ui, "tint", &mut g.tint, -1.0..=1.0, 0.0, "");
                slider(ui, "contrast", &mut g.contrast, 0.0..=3.0, 1.0, "");
                slider(ui, "saturation", &mut g.saturation, 0.0..=3.0, 1.0, "");
                slider(ui, "lift", &mut g.lift, -0.2..=0.2, 0.0, "");
                slider(ui, "gamma", &mut g.gamma, 0.2..=3.0, 1.0, "");
                slider(ui, "gain", &mut g.gain, 0.0..=4.0, 1.0, "");
            });
            match head {
                Head::Remove => self.set_grade_quiet(t, i, None, false),
                Head::Reset => self.set_grade_quiet(t, i, Some(model::Grade::default()), false),
                Head::Keep if g != g0 => self.set_grade_quiet(t, i, Some(g), true),
                Head::Keep => {}
            }
        }
    }

    /// A transform from the panel: no words over the viewer (a drag would
    /// flood them), one undo step once it settles.
    fn set_reframe_quiet(&mut self, t: usize, i: usize, r: model::Reframe) {
        let start = self.rec.seq.tracks[t].clips[i].start;
        self.rec.seq.set_reframe(t, i, r);
        self.rec.sel = Selection::Clip { track: t, start };
        self.changed(true);
    }

    /// A text clip's animators: "+ add effect > text animation" (the
    /// presets) and a section per animator.
    fn text_anim_controls(&mut self, ui: &mut egui::Ui, t: usize, i: usize, anims0: &[TextAnim]) {
        let mut anims = anims0.to_vec();
        let mut burst = true;
        ui.separator();
        ui.menu_button("+ add effect", |ui| {
            ui.menu_button("text animation", |ui| {
                for p in model::textanim::PRESETS {
                    if ui.button(p).clicked() {
                        anims.push(TextAnim::preset(p));
                        burst = false;
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("blank").clicked() {
                    anims.push(TextAnim::default());
                    burst = false;
                    ui.close();
                }
            });
        });
        let mut remove = None;
        for (k, a) in anims.iter_mut().enumerate() {
            ui.separator();
            ui.push_id(("text-anim", k), |ui| {
                let mut on = a.on;
                match heading(ui, &a.name, Some(&mut on)) {
                    Head::Reset => {
                        *a = TextAnim::preset(&a.name.clone());
                        burst = false;
                    }
                    Head::Remove => remove = Some(k),
                    Head::Keep => a.on = on,
                }
                ui.add_enabled_ui(a.on, |ui| text_anim_rows(ui, a));
            });
        }
        if let Some(k) = remove {
            anims.remove(k);
            burst = false;
        }
        if anims != anims0 {
            let start = self.rec.seq.tracks[t].clips[i].start;
            self.rec.seq.set_text_anims(t, i, anims);
            self.rec.sel = Selection::Clip { track: t, start };
            self.changed(burst);
        }
    }

    pub(crate) fn set_grade_quiet(&mut self, t: usize, i: usize, g: Option<model::Grade>, burst: bool) {
        let start = self.rec.seq.tracks[t].clips[i].start;
        self.rec.seq.set_grade(t, i, g);
        self.rec.sel = Selection::Clip { track: t, start };
        self.changed(burst);
    }
}

/// `slider` for an f32.
fn slider32(ui: &mut egui::Ui, label: &str, v: &mut f32, range: std::ops::RangeInclusive<f64>, neutral: f64, suffix: &str) -> bool {
    let mut x = *v as f64;
    let moved = slider(ui, label, &mut x, range, neutral, suffix);
    if moved {
        *v = x as f32;
    }
    moved
}

/// A frame count as a slider.
fn frames(ui: &mut egui::Ui, label: &str, v: &mut u32) {
    let mut x = *v as f64;
    if slider(ui, label, &mut x, 0.0..=120.0, 0.0, " f") {
        *v = x.round().clamp(0.0, 100_000.0) as u32;
    }
}

/// One animator's rows: what it counts, when it moves, which units, and
/// what it does to them.
fn text_anim_rows(ui: &mut egui::Ui, a: &mut TextAnim) {
    ui.horizontal(|ui| {
        ui.add_sized([74.0, 18.0], egui::Label::new("by"));
        for u in TextUnit::ALL {
            ui.selectable_value(&mut a.unit, u, u.name());
        }
    });
    frames(ui, "in", &mut a.in_frames);
    frames(ui, "out", &mut a.out_frames);
    ui.horizontal(|ui| {
        ui.add_sized([74.0, 18.0], egui::Label::new("ease"));
        egui::ComboBox::from_id_salt("ease").selected_text(a.ease.name()).show_ui(ui, |ui| {
            for e in Ease::ALL {
                ui.selectable_value(&mut a.ease, e, e.name());
            }
        });
    });
    if a.hold() {
        // No sweeps: the window selects all the time.
        ui.label(RichText::new("no in / out: holds over the window below").weak().small());
        ui.horizontal(|ui| {
            ui.add_sized([74.0, 18.0], egui::Label::new("shape"));
            egui::ComboBox::from_id_salt("shape").selected_text(a.shape.name()).show_ui(ui, |ui| {
                for s in RangeShape::ALL {
                    ui.selectable_value(&mut a.shape, s, s.name());
                }
            });
        });
        slider32(ui, "start", &mut a.start, 0.0..=100.0, 0.0, "%");
        slider32(ui, "end", &mut a.end, 0.0..=100.0, 100.0, "%");
        slider32(ui, "offset", &mut a.offset, -100.0..=100.0, 0.0, "%");
    }
    slider32(ui, "softness", &mut a.softness, 0.0..=100.0, 40.0, "%");
    slider32(ui, "amount", &mut a.amount, -100.0..=100.0, 100.0, "%");
    ui.horizontal(|ui| {
        ui.add_sized([74.0, 18.0], egui::Label::new("order"));
        ui.checkbox(&mut a.random, "random");
        if a.random {
            ui.add(egui::DragValue::new(&mut a.seed).range(0..=9999).prefix("seed "));
        }
    });
    ui.label(RichText::new("from").weak().small()).on_hover_text("what a unit is, fully driven: where it comes from (in) or goes to (out)");
    slider32(ui, "position x", &mut a.position[0], -4.0..=4.0, 0.0, " em");
    slider32(ui, "position y", &mut a.position[1], -4.0..=4.0, 0.0, " em");
    let mut sc = a.scale * 100.0;
    if slider32(ui, "scale", &mut sc, 0.0..=400.0, 100.0, "%") {
        a.scale = (sc / 100.0).clamp(0.0, 20.0);
    }
    slider32(ui, "rotation", &mut a.rotation, -360.0..=360.0, 0.0, "°");
    slider32(ui, "tracking", &mut a.tracking, -200.0..=1000.0, 0.0, "");
    let mut op = a.opacity * 100.0;
    if slider32(ui, "opacity", &mut op, 0.0..=100.0, 100.0, "%") {
        a.opacity = (op / 100.0).clamp(0.0, 1.0);
    }
    ui.horizontal(|ui| {
        ui.add_sized([74.0, 18.0], egui::Label::new("tint"));
        ui.color_edit_button_srgba_unmultiplied(&mut a.tint).on_hover_text("alpha is how much of it");
    });
}
