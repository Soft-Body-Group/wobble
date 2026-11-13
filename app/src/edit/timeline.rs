//! The timeline, bin and clip strip — NLE gestures over `EditState`.
//!
//! Ported from the standalone scrub app. Nothing in here touches the host:
//! moving the playhead records a `seek_request`, placing a clip records a
//! `new_clips` entry, and so on; `Gfx` applies those each frame.

use crate::theme;
use super::cache::CacheState;
use super::model::{self, Clip, Media};
use super::thumbs::Thumb;
use super::{BinDrag, EditState, Insert, TlDrag, TlHit, TrimSide};

pub fn input_combo(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash + std::fmt::Debug,
    clip: &mut Clip,
) -> bool {
    use engine::color::Colorspace;
    let guess = model::default_input(&clip.media().extension());
    let auto_label = format!(
        "auto ({})",
        guess.map(model::input_label).unwrap_or_else(|| "working".into())
    );
    let current = clip.input.map(model::input_label).unwrap_or_else(|| auto_label.clone());
    let mut changed = false;
    egui::ComboBox::from_id_salt(id_salt).width(150.0).selected_text(current).show_ui(ui, |ui| {
        if ui.selectable_label(clip.input.is_none(), &auto_label).clicked()
            && clip.input.take().is_some()
        {
            changed = true;
        }
        for (label, cs) in Colorspace::PRESETS {
            if ui.selectable_label(clip.input == Some(cs), label).clicked()
                && clip.input != Some(cs)
            {
                clip.input = Some(cs);
                changed = true;
            }
        }
    });
    changed
}

fn badge_colour(clip: &Clip, state: Option<&CacheState>) -> Option<(egui::Color32, String)> {
    if clip.is_blank() {
        return Some((theme::info(), "blank — a wobble graph, no footage".into()));
    }
    if matches!(clip.media(), Media::Movie { .. }) {
        return Some((theme::info(), "movie — plays direct".into()));
    }
    Some(match state? {
        CacheState::Ready(_) => (theme::ok(), "proxy ready".into()),
        CacheState::Encoding(p) => {
            (theme::warn(), format!("proxy {:.0}%", p * 100.0))
        }
        CacheState::Failed(e) => (theme::error(), e.clone()),
    })
}

impl EditState {
    pub fn bin_ui(&mut self, ui: &mut egui::Ui) {
        self.thumbs.poll(ui.ctx());
        self.want_posters();
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("bin").strong());
            ui.label(egui::RichText::new("· drag :: to timeline or onto a folder · double-click the poster to view").weak().size(11.0));
        });
        ui.horizontal(|ui| {
            if ui
                .button("add sequences…")
                .on_hover_text(if self.seq_manip_missing {
                    "opens wobble's footage browser — recursive, picks land here \
                     (sequence manipulator isn't installed)"
                } else {
                    "opens sequence manipulator if it's installed (its recursive search finds the \
                     footage; right-click sequences there → Send to wobble edit), \
                     else wobble's own footage browser"
                })
                .clicked()
            {
                self.launch_seq_manip();
            }
            if ui
                .small_button("blank clip")
                .on_hover_text(
                    "a clip with no footage: its shot starts as an empty wobble graph \
                     (100 frames; drag its out point on the timeline to change that)",
                )
                .clicked()
            {
                self.add_blank_clip();
            }
            if ui
                .small_button("browse…")
                .on_hover_text("wobble's own footage browser (also R) — recursive, picks land here")
                .clicked()
            {
                self.browse_request = true;
            }
            if ui
                .small_button("file…")
                .on_hover_text("point at one exact frame or movie")
                .clicked()
            {
                if let Some(f) = crate::filedialog::FileDialog::new()
                    .add_filter("footage", &["exr", "dpx", "png", "jpg", "mov", "mp4"])
                    .pick_file()
                {
                    self.add_clip_from_sample(f);
                }
            }
            if ui.button("new folder").clicked() {
                self.new_folder = Some(String::new());
            }
            ui.toggle_value(&mut self.big_thumbs, "big").on_hover_text("larger posters");
        });
        if let Some(name) = self.new_folder.as_mut() {
            let mut done = false;
            let mut cancel = false;
            ui.horizontal(|ui| {
                let r = ui.add(egui::TextEdit::singleline(name).hint_text("folder name").desired_width(140.0));
                // Enter gives the focus up: read that BEFORE taking it back,
                // or `lost_focus` is false by the time it is asked.
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    done = true;
                } else {
                    r.request_focus();
                }
                if ui.small_button("add").clicked() {
                    done = true;
                }
                if ui.small_button("×").clicked() || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    cancel = true;
                }
            });
            if done {
                let n = name.trim().to_string();
                if !n.is_empty() && !self.project.folders.contains(&n) {
                    self.project.folders.push(n);
                }
                self.new_folder = None;
            } else if cancel {
                self.new_folder = None;
            }
        }
        ui.separator();

        let mut place: Option<u64> = None;
        let mut remove: Option<u64> = None;
        let mut view: Option<u64> = None;
        let mut duplicate: Option<u64> = None;
        let mut moved: Option<(u64, String)> = None;
        let mut drop_folder: Option<String> = None;
        let mut rebind: Option<u64> = None;
        let folders: Vec<String> = self.project.folders.clone();
        let thumb_w = if self.big_thumbs { 192.0 } else { 96.0 };

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Root clips, then one collapsible section per folder.
            let mut sections: Vec<Option<String>> = vec![None];
            sections.extend(folders.iter().cloned().map(Some));
            for section in sections {
                let in_section = |c: &Clip| match &section {
                    None => c.folder.is_empty(),
                    Some(f) => &c.folder == f,
                };
                let count = self.project.bin.iter().filter(|c| in_section(c)).count();
                let mut open = true;
                if let Some(f) = &section {
                    let collapsed = self.collapsed.contains(f);
                    let header = ui.add(
                        egui::Button::new(
                            egui::RichText::new(format!(
                                "{} {}  ({count})",
                                if collapsed { "▸" } else { "▾" },
                                f
                            ))
                            .strong(),
                        )
                        .frame(false),
                    );
                    if header.clicked() {
                        if collapsed {
                            self.collapsed.remove(f);
                        } else {
                            self.collapsed.insert(f.clone());
                        }
                    }
                    if header.dnd_hover_payload::<BinDrag>().is_some() {
                        ui.painter().rect_stroke(
                            header.rect.expand(2.0),
                            3.0,
                            egui::Stroke::new(1.5, theme::timeline_active()),
                            egui::StrokeKind::Outside,
                        );
                    }
                    if let Some(p) = header.dnd_release_payload::<BinDrag>() {
                        moved = Some((p.0, f.clone()));
                    }
                    header.context_menu(|ui| {
                        if ui.button("delete folder (clips move to root)").clicked() {
                            drop_folder = Some(f.clone());
                            ui.close();
                        }
                    });
                    open = !collapsed;
                } else if count == 0 && !folders.is_empty() {
                    continue;
                }
                if !open {
                    continue;
                }
                for i in 0..self.project.bin.len() {
                    if !in_section(&self.project.bin[i]) {
                        continue;
                    }
                    let uid = self.project.bin[i].uid;
                    let state = self.cache_state(uid);
                    let viewing = self.source == Some(uid);
                    let frame = egui::Frame::group(ui.style()).inner_margin(6.0);
                    let frame = if viewing {
                        frame.stroke(egui::Stroke::new(1.5, theme::timeline_source()))
                    } else {
                        frame
                    };
                    frame
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            let clip = &mut self.project.bin[i];
                            ui.horizontal(|ui| {
                                // Poster.
                                let size = egui::vec2(thumb_w, thumb_w * 9.0 / 16.0);
                                match self.thumbs.have.get(&uid) {
                                    _ if clip.is_blank() => {
                                        let (r, resp) = ui.allocate_exact_size(size, egui::Sense::click());
                                        if resp.double_clicked() {
                                            view = Some(uid);
                                        }
                                        resp.on_hover_text("a blank wobble clip — double-click to view");
                                        ui.painter().rect_filled(r, 2.0, theme::well());
                                        ui.painter().text(
                                            r.center(),
                                            egui::Align2::CENTER_CENTER,
                                            "wobble",
                                            egui::FontId::proportional(11.0),
                                            theme::text_weak(),
                                        );
                                    }
                                    Some(Thumb::Ready(tex)) => {
                                        let aspect = tex.size()[0] as f32 / tex.size()[1].max(1) as f32;
                                        let sz = if aspect >= size.x / size.y {
                                            egui::vec2(size.x, size.x / aspect)
                                        } else {
                                            egui::vec2(size.y * aspect, size.y)
                                        };
                                        let (r, resp) = ui.allocate_exact_size(size, egui::Sense::click());
                                        if resp.double_clicked() {
                                            view = Some(uid);
                                        }
                                        resp.on_hover_text("double-click to view in the source viewer");
                                        let img = egui::Rect::from_center_size(r.center(), sz);
                                        ui.painter().rect_filled(r, 2.0, theme::well());
                                        ui.painter().image(
                                            tex.id(),
                                            img,
                                            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                                            egui::Color32::WHITE,
                                        );
                                    }
                                    Some(Thumb::Failed(e)) => {
                                        let (r, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                                        ui.painter().rect_filled(r, 2.0, theme::well());
                                        ui.painter().text(
                                            r.center(),
                                            egui::Align2::CENTER_CENTER,
                                            "no poster",
                                            egui::FontId::proportional(10.0),
                                            theme::text_weak(),
                                        );
                                        ui.allocate_rect(r, egui::Sense::hover()).on_hover_text(e);
                                    }
                                    _ => {
                                        let (r, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                                        ui.painter().rect_filled(r, 2.0, theme::well());
                                        ui.put(r, egui::Spinner::new().size(14.0));
                                    }
                                }
                                ui.vertical(|ui| {
                                    ui.horizontal(|ui| {
                                        let handle = ui.dnd_drag_source(
                                            egui::Id::new(("bin-drag", uid)),
                                            BinDrag(uid),
                                            |ui| {
                                                ui.label(egui::RichText::new("::").strong().monospace())
                                                    .on_hover_text("drag onto the timeline, or onto a folder");
                                            },
                                        );
                                        self.handle_rects.insert(uid, handle.response.rect);
                                        ui.label(egui::RichText::new(&clip.name).strong());
                                        if clip.latest() != clip.current {
                                            ui.label(egui::RichText::new("⬆").color(theme::timeline_active()))
                                                .on_hover_text("newer version on disk");
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        if clip.is_blank() {
                                            ui.label(
                                                egui::RichText::new(format!("blank · {}f", clip.len())).weak().small(),
                                            );
                                            return;
                                        }
                                        let n_versions = clip.versions.len();
                                        let mut pick = clip.current;
                                        egui::ComboBox::from_id_salt(("bin-ver", uid))
                                            .width(70.0)
                                            .selected_text(clip.version().label.clone())
                                            .show_ui(ui, |ui| {
                                                for v in 0..n_versions {
                                                    ui.selectable_value(&mut pick, v, clip.versions[v].label.clone());
                                                }
                                            });
                                        if pick != clip.current {
                                            clip.set_version(pick);
                                            rebind = Some(uid);
                                        }
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "{}f · in {} out {}",
                                                clip.media().len(),
                                                clip.in_off,
                                                clip.out_off
                                            ))
                                            .weak()
                                            .small(),
                                        );
                                    });
                                    ui.horizontal(|ui| {
                                        if ui
                                            .small_button(if viewing { "■ viewing" } else { "▶ view" })
                                            .on_hover_text("play this clip in the viewer without adding it to the timeline")
                                            .clicked()
                                        {
                                            view = Some(uid);
                                        }
                                        if ui.small_button("+").on_hover_text("append to timeline").clicked() {
                                            place = Some(uid);
                                        }
                                        if ui
                                            .small_button("⧉")
                                            .on_hover_text(
                                                "duplicate: a second bin entry on the same media, \
                                                 with its own wobble graph",
                                            )
                                            .clicked()
                                        {
                                            duplicate = Some(uid);
                                        }
                                        if ui.small_button("×").on_hover_text("remove from bin (and its clips + graph)").clicked() {
                                            remove = Some(uid);
                                        }
                                        if !folders.is_empty() {
                                            let cur = if clip.folder.is_empty() { "root".to_string() } else { clip.folder.clone() };
                                            egui::ComboBox::from_id_salt(("bin-folder", uid))
                                                .width(90.0)
                                                .selected_text(cur)
                                                .show_ui(ui, |ui| {
                                                    if ui.selectable_label(clip.folder.is_empty(), "root").clicked() {
                                                        moved = Some((uid, String::new()));
                                                    }
                                                    for f in &folders {
                                                        if ui.selectable_label(&clip.folder == f, f).clicked() {
                                                            moved = Some((uid, f.clone()));
                                                        }
                                                    }
                                                });
                                        }
                                        if let Some((color, tip)) = badge_colour(clip, state.as_ref()) {
                                            let (dot, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                                            ui.painter().circle_filled(dot.center(), 3.5, color);
                                            ui.allocate_rect(dot, egui::Sense::hover()).on_hover_text(tip);
                                        }
                                    });
                                    ui.horizontal(|ui| {
                                        if input_combo(ui, ("bin-input", uid), clip) {
                                            rebind = Some(uid);
                                        }
                                    });
                                });
                            });
                    });
                    // NOTE: no row-wide click target here. One used to sit
                    // over the whole group for double-click-to-view, and
                    // because it was added last it was ON TOP of the row's
                    // own buttons and swallowed their clicks — the bin
                    // looked frozen. The POSTER is the double-click target.
                }
            }
            if self.project.bin.is_empty() {
                ui.label(egui::RichText::new("empty — browse…, or send sequences here").weak());
            }
        });
        if let Some(uid) = place {
            self.append_from_bin(uid);
        }
        if let Some(uid) = remove {
            self.remove_bin(uid);
        }
        if let Some(uid) = duplicate {
            if let Some(src) = self.project.bin.iter().find(|c| c.uid == uid) {
                let dup = src.duplicate_master();
                self.project.bin.push(dup);
            }
        }
        if let Some(uid) = view {
            self.source_request = Some(if self.source == Some(uid) { None } else { Some(uid) });
        }
        if let Some((uid, folder)) = moved {
            if let Some(c) = self.project.bin.iter_mut().find(|c| c.uid == uid) {
                c.folder = folder;
            }
        }
        if let Some(f) = drop_folder {
            for c in self.project.bin.iter_mut() {
                if c.folder == f {
                    c.folder.clear();
                }
            }
            self.project.folders.retain(|x| *x != f);
            self.collapsed.remove(&f);
        }
        if let Some(uid) = rebind {
            // A bin clip's version or input changed: its poster and — if it
            // is being viewed — its source graph are stale.
            self.thumbs.invalidate(uid);
            if self.source == Some(uid) {
                self.source_request = Some(Some(uid));
            }
        }
    }

    /// The SOURCE VIEWER's strip: the clip's whole media as a ruler with
    /// in/out marks, the transport tick, and the ways off it — mark, insert,
    /// back to the timeline.
    pub fn source_ui(&mut self, ui: &mut egui::Ui, tick: u64, playing: &mut bool) {
        let Some(uid) = self.source else { return };
        let Some(idx) = self.project.bin.iter().position(|c| c.uid == uid) else {
            self.source_request = Some(None);
            return;
        };
        let n = self.project.bin[idx].media().len().max(1);
        let tick = (tick as usize).min(n - 1);
        // I/O mark wherever the pointer is — an editor reaches for them
        // while watching the picture, not while hovering the strip.
        if ui.memory(|m| m.focused()).is_none() {
            let (i_key, o_key) = ui.input(|i| {
                (i.key_pressed(egui::Key::I), i.key_pressed(egui::Key::O))
            });
            let c = &mut self.project.bin[idx];
            if i_key {
                c.in_off = tick.min(c.out_off);
            }
            if o_key {
                c.out_off = tick.max(c.in_off);
            }
        }
        let (name, ver) = {
            let c = &self.project.bin[idx];
            (c.name.clone(), c.version().label.clone())
        };
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("SOURCE").strong().color(theme::timeline_source()));
            ui.label(egui::RichText::new(format!("{name} {ver}")).strong());
            ui.monospace(format!("f{tick:04} / {n}"));
            let c = &mut self.project.bin[idx];
            if ui.small_button("I").on_hover_text("mark in here (I)").clicked() {
                c.in_off = tick.min(c.out_off);
            }
            if ui.small_button("O").on_hover_text("mark out here (O)").clicked() {
                c.out_off = tick.max(c.in_off);
            }
            if ui.small_button("clear").on_hover_text("whole clip").clicked() {
                c.in_off = 0;
                c.out_off = n - 1;
            }
            ui.label(egui::RichText::new(format!("in {} · out {} · {}f", c.in_off, c.out_off, c.len())).weak());
            ui.separator();
            if ui
                .button("insert at playhead")
                .on_hover_text("place the marked range on the timeline at the playhead")
                .clicked()
            {
                self.insert_request = Some(Insert::AtPlayhead);
            }
            if ui.button("append").on_hover_text("place the marked range after the last clip").clicked() {
                self.insert_request = Some(Insert::Append);
            }
            ui.separator();
            if ui.button("← timeline").on_hover_text("back to the edit").clicked() {
                self.source_request = Some(None);
            }
        });
        // The ruler.
        let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 26.0), egui::Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, theme::well());
        let ppf = rect.width() / n as f32;
        let x_of = |f: usize| rect.left() + f as f32 * ppf;
        let c = &mut self.project.bin[idx];
        let marked = egui::Rect::from_min_max(
            egui::pos2(x_of(c.in_off), rect.top() + 2.0),
            egui::pos2(x_of(c.out_off + 1), rect.bottom() - 2.0),
        );
        painter.rect_filled(marked, 2.0, theme::timeline_source().gamma_multiply(0.35));
        for (x, _) in [(x_of(c.in_off), "in"), (x_of(c.out_off + 1), "out")] {
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(2.0, theme::timeline_source()),
            );
        }
        let px = x_of(tick) + ppf * 0.5;
        painter.line_segment(
            [egui::pos2(px, rect.top()), egui::pos2(px, rect.bottom())],
            egui::Stroke::new(1.5, theme::timeline_playhead()),
        );
        // Drag near a mark moves the mark; anywhere else scrubs.
        if response.drag_started() {
            let o = ui.input(|i| i.pointer.press_origin()).unwrap_or(rect.left_top());
            self.source_grab = if (o.x - x_of(c.in_off)).abs() <= 6.0 {
                Some(TrimSide::In)
            } else if (o.x - x_of(c.out_off + 1)).abs() <= 6.0 {
                Some(TrimSide::Out)
            } else {
                None
            };
        }
        if response.dragged() || response.clicked() {
            if let Some(p) = response.interact_pointer_pos() {
                let f = (((p.x - rect.left()) / ppf).floor().max(0.0) as usize).min(n - 1);
                match self.source_grab {
                    Some(TrimSide::In) if response.dragged() => c.in_off = f.min(c.out_off),
                    Some(TrimSide::Out) if response.dragged() => c.out_off = f.max(c.in_off),
                    _ => {
                        self.source_seek = Some(f as u64);
                        *playing = false;
                    }
                }
            }
        }
        if response.drag_stopped() {
            self.source_grab = None;
        }
        ui.add_space(2.0);
    }

    /// The selected clip's strip: version, trim, input, remove.
    pub fn clip_strip_ui(&mut self, ui: &mut egui::Ui) {
        let Some(sel) = self.selected.filter(|s| *s < self.project.clips.len()) else { return };
        let mut rebind = false;
        let mut jump: Option<f64> = None;
        let mut remove = false;
        let mut import: Option<std::path::PathBuf> = None;
        let uid = self.project.clips[sel].uid;
        let state = self.cache_state(uid);
        let start = self.project.clips[sel].start;
        let clip = &mut self.project.clips[sel];
        let blank = clip.is_blank();
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(&clip.name).strong());
            if !blank {
                let n_versions = clip.versions.len();
                let mut pick = clip.current;
                egui::ComboBox::from_id_salt(("ver", sel))
                    .selected_text(clip.version().label.clone())
                    .show_ui(ui, |ui| {
                        for i in 0..n_versions {
                            let label = format!(
                                "{}  ({} frames)",
                                clip.versions[i].label,
                                clip.versions[i].media.len()
                            );
                            ui.selectable_value(&mut pick, i, label);
                        }
                    });
                if pick != clip.current {
                    clip.set_version(pick);
                    rebind = true;
                }
                let latest = clip.latest();
                if latest != clip.current
                    && ui
                        .button(
                            egui::RichText::new(format!("⬆ {}", clip.versions[latest].label))
                                .color(theme::timeline_active()),
                        )
                        .on_hover_text("newer version on disk — click to bump")
                        .clicked()
                {
                    clip.set_version(latest);
                    rebind = true;
                }
                if ui.button("rescan").clicked() {
                    clip.rescan();
                    rebind = true;
                }
                ui.separator();
            }
            let max = clip.media().len() - 1;
            let (mut i0, mut i1) = (clip.in_off, clip.out_off);
            ui.label("in");
            ui.add(egui::DragValue::new(&mut i0).update_while_editing(false).range(0..=max));
            ui.label("out");
            ui.add(egui::DragValue::new(&mut i1).update_while_editing(false).range(0..=max));
            if i0 != clip.in_off || i1 != clip.out_off {
                clip.in_off = i0.min(i1);
                clip.out_off = i1.max(i0);
                jump = Some(start as f64);
            }
            ui.separator();
            if !blank && input_combo(ui, ("clip-input", uid), clip) {
                rebind = true;
            }
            ui.separator();
            ui.label(format!("{} frames", clip.len()));
            if let Some((color, tip)) = badge_colour(clip, state.as_ref()) {
                ui.colored_label(color, tip);
            }
            ui.separator();
            if ui
                .button("import script…")
                .on_hover_text("use an existing wobble script as this clip's shot (the edit takes its own copy)")
                .clicked()
            {
                if let Some(f) = crate::filedialog::FileDialog::new().add_filter("wobble script", &["json"]).pick_file() {
                    import = Some(f);
                }
            }
            if ui.button("remove clip").clicked() {
                remove = true;
            }
        });
        if rebind {
            self.rebind.push(uid);
        }
        if let Some(p) = import {
            self.import_requests.push((uid, p));
        }
        if let Some(f) = jump {
            self.set_playhead(f);
        }
        if remove {
            self.remove_clip(sel);
        }
    }

    pub fn timeline_ui(&mut self, ui: &mut egui::Ui, playing: &mut bool) {
        // Delete/backspace lifts the selected clip off the timeline. Safe to
        // read window-wide: in edit mode this window has no node graph, and
        // a focused text field (folder name) keeps its own keys.
        if ui.memory(|m| m.focused()).is_none() {
            let del = ui.input(|i| {
                i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)
            });
            if del {
                if let Some(sel) = self.selected.filter(|s| *s < self.project.clips.len()) {
                    self.remove_clip(sel);
                }
            }
        }
        let total = self.project.total_len();
        ui.add_space(4.0);
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 64.0), egui::Sense::click_and_drag());
        self.tl_rect = Some(rect);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, theme::well());

        let hover_payload = response.dnd_hover_payload::<BinDrag>();
        let release_payload = response.dnd_release_payload::<BinDrag>();
        let pointer = ui.ctx().pointer_latest_pos();

        // Zoom keys, only while the pointer is over the strip: wobble's own
        // hotkeys own the rest of the window.
        if response.hovered() {
            let view_w = rect.width();
            ui.input(|i| {
                if i.key_pressed(egui::Key::F) {
                    self.tl_zoom = None;
                    self.tl_scroll = 0.0;
                }
                if i.key_pressed(egui::Key::Minus) {
                    if let Some(z) = self.tl_zoom {
                        self.zoom_anchored(z / 1.5, view_w);
                    }
                }
                if i.key_pressed(egui::Key::Equals) {
                    if let Some(z) = self.tl_zoom {
                        self.zoom_anchored(z * 1.5, view_w);
                    }
                }
            });
        }

        if total == 0 {
            if let Some(p) = release_payload {
                self.place_from_bin(p.0, 0);
            }
            if hover_payload.is_some() {
                painter.rect_stroke(
                    rect.shrink(1.0),
                    4.0,
                    egui::Stroke::new(1.5, theme::timeline_active()),
                    egui::StrokeKind::Inside,
                );
            }
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "drop clips from the bin",
                egui::FontId::proportional(12.0),
                theme::text_weak(),
            );
            self.transport_row(ui, playing, 1.0, rect.width());
            return;
        }
        let fit_ppf = (rect.width() / total as f32).min(40.0);
        let mut ppf = *self.tl_zoom.get_or_insert(fit_ppf);
        if response.hovered() {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta, i.zoom_delta()));
            if zoom != 1.0 {
                if let Some(p) = pointer {
                    let anchor = self.tl_scroll + (p.x - rect.left()) / ppf;
                    ppf = (ppf * zoom).clamp(0.02, 200.0);
                    self.tl_scroll = anchor - (p.x - rect.left()) / ppf;
                    self.tl_zoom = Some(ppf);
                }
            }
            if scroll.x != 0.0 || scroll.y != 0.0 {
                self.tl_scroll -= (scroll.x + scroll.y) / ppf;
            }
        }
        let max_scroll = (total as f32 - rect.width() / ppf).max(0.0);
        self.tl_scroll = self.tl_scroll.clamp(0.0, max_scroll);
        if *playing {
            let vis = rect.width() / ppf;
            let ph = self.playhead as f32;
            if ph < self.tl_scroll || ph > self.tl_scroll + vis {
                self.tl_scroll = (ph - vis * 0.1).clamp(0.0, max_scroll);
            }
        }
        let origin_x = rect.left() - self.tl_scroll * ppf;

        let ruler = egui::Rect::from_min_max(rect.left_top(), egui::pos2(rect.right(), rect.top() + 14.0));
        let blocks_top = rect.top() + 16.0;
        let hbar_top = rect.bottom() - 8.0;
        let block_of = |x: f32, len: usize| {
            egui::Rect::from_min_size(
                egui::pos2(x, blocks_top),
                egui::vec2((len as f32 * ppf - 2.0).max(1.0), hbar_top - 4.0 - blocks_top),
            )
        };
        let blocks: Vec<egui::Rect> = self
            .project
            .clips
            .iter()
            .map(|c| block_of(origin_x + c.start as f32 * ppf, c.len()))
            .collect();
        let adjacent: Vec<bool> = (0..self.project.clips.len().saturating_sub(1))
            .map(|i| self.project.clips[i + 1].start == self.project.clips[i].end())
            .collect();

        const EDGE: f32 = 6.0;
        const ROLL: f32 = 3.0;
        let playhead_px = origin_x + self.playhead as f32 * ppf;
        let hit = |p: egui::Pos2| -> TlHit {
            if (p.x - playhead_px).abs() <= 5.0 && p.y < hbar_top {
                return TlHit::Playhead;
            }
            if p.y >= hbar_top {
                return TlHit::HBar;
            }
            if p.y < blocks_top {
                return TlHit::Ruler;
            }
            let n = blocks.len();
            for i in 0..n.saturating_sub(1) {
                if !adjacent[i] {
                    continue;
                }
                let cut_x = blocks[i + 1].left();
                if (p.x - cut_x).abs() <= ROLL && p.y >= blocks_top && p.y <= blocks[i].bottom() + 6.0 {
                    return TlHit::Roll(i);
                }
            }
            for (i, b) in blocks.iter().enumerate() {
                if !b.expand2(egui::vec2(1.0, 6.0)).contains(p) || b.width() < 18.0 {
                    continue;
                }
                if p.x - b.left() <= EDGE {
                    return TlHit::Trim(i, TrimSide::In);
                }
                if b.right() - p.x <= EDGE {
                    return TlHit::Trim(i, TrimSide::Out);
                }
                return TlHit::Body(i);
            }
            TlHit::Empty
        };

        if response.drag_started() {
            let origin = ui.input(|i| i.pointer.press_origin()).or(pointer);
            self.tl_drag = origin.map(|o| match hit(o) {
                TlHit::Trim(i, side) => {
                    let c = &self.project.clips[i];
                    let (cs, ce) = (c.start, c.end());
                    let prev_end = self
                        .project
                        .clips
                        .iter()
                        .enumerate()
                        .filter(|(j, x)| *j != i && x.end() <= cs)
                        .map(|(_, x)| x.end())
                        .max()
                        .unwrap_or(0);
                    let next_start = self
                        .project
                        .clips
                        .iter()
                        .enumerate()
                        .filter(|(j, x)| *j != i && x.start >= ce)
                        .map(|(_, x)| x.start)
                        .min()
                        .unwrap_or(usize::MAX);
                    TlDrag::Trim {
                        clip: i,
                        side,
                        start_in: c.in_off,
                        start_out: c.out_off,
                        start_pos: c.start,
                        prev_end,
                        next_start,
                        start_x: o.x,
                        ppf,
                    }
                }
                TlHit::Roll(i) => TlDrag::Roll {
                    left: i,
                    start_out: self.project.clips[i].out_off,
                    start_in: self.project.clips[i + 1].in_off,
                    b_start: self.project.clips[i + 1].start,
                    start_x: o.x,
                    ppf,
                },
                TlHit::Body(i) => {
                    if ui.input(|inp| inp.modifiers.alt) {
                        let c = &self.project.clips[i];
                        TlDrag::Slip { clip: i, start_in: c.in_off, start_out: c.out_off, start_x: o.x, ppf }
                    } else {
                        TlDrag::Move {
                            clip: i,
                            grab: ((o.x - origin_x) / ppf) as i64 - self.project.clips[i].start as i64,
                        }
                    }
                }
                TlHit::HBar => TlDrag::HBar,
                TlHit::Playhead | TlHit::Ruler | TlHit::Empty => TlDrag::Scrub,
            });
        }

        if response.hovered() || self.tl_drag.is_some() {
            let icon = match &self.tl_drag {
                Some(TlDrag::Trim { .. }) => Some(egui::CursorIcon::ResizeHorizontal),
                Some(TlDrag::Roll { .. }) => Some(egui::CursorIcon::ResizeColumn),
                Some(TlDrag::Move { .. }) => Some(egui::CursorIcon::Grabbing),
                Some(TlDrag::Slip { .. }) => Some(egui::CursorIcon::ResizeHorizontal),
                Some(TlDrag::HBar) => Some(egui::CursorIcon::Grab),
                Some(TlDrag::Scrub) => None,
                None => pointer.and_then(|p| match hit(p) {
                    TlHit::Trim(..) => Some(egui::CursorIcon::ResizeHorizontal),
                    TlHit::Roll(_) => Some(egui::CursorIcon::ResizeColumn),
                    TlHit::Body(_) => {
                        if ui.input(|inp| inp.modifiers.alt) {
                            Some(egui::CursorIcon::ResizeHorizontal)
                        } else {
                            Some(egui::CursorIcon::Grab)
                        }
                    }
                    TlHit::Playhead => Some(egui::CursorIcon::ResizeHorizontal),
                    TlHit::HBar => Some(egui::CursorIcon::Grab),
                    _ => None,
                }),
            };
            if let Some(icon) = icon {
                ui.output_mut(|o| o.cursor_icon = icon);
            }
        }

        let mut trim_overlay: Option<usize> = None;
        let mut slip_overlay: Option<usize> = None;
        let mut roll_overlay: Option<usize> = None;
        let mut move_ghost: Option<(usize, usize)> = None;
        let mut edited_clip: Option<usize> = None;
        match (&self.tl_drag, pointer) {
            (
                Some(TlDrag::Trim { clip, side, start_in, start_out, start_pos, prev_end, next_start, start_x, ppf: p0 }),
                Some(p),
            ) if response.dragged() => {
                let d = ((p.x - start_x) / p0.max(1e-6)).round() as i64;
                let c = &mut self.project.clips[*clip];
                let n = c.media().len() as i64;
                match side {
                    TrimSide::In => {
                        let d = d
                            .clamp(*prev_end as i64 - *start_pos as i64, i64::MAX)
                            .clamp(-(*start_in as i64).min(i64::MAX), *start_out as i64 - *start_in as i64);
                        c.in_off = (*start_in as i64 + d) as usize;
                        c.start = (*start_pos as i64 + d) as usize;
                    }
                    TrimSide::Out => {
                        let max_by_next = if *next_start == usize::MAX {
                            i64::MAX
                        } else {
                            *next_start as i64 - *start_pos as i64 + *start_in as i64 - 1
                        };
                        c.out_off =
                            (*start_out as i64 + d).clamp(*start_in as i64, (n - 1).min(max_by_next)) as usize;
                    }
                }
                trim_overlay = Some(*clip);
                edited_clip = Some(*clip);
            }
            (Some(TlDrag::Roll { left, start_out, start_in, b_start, start_x, ppf: p0 }), Some(p))
                if response.dragged() =>
            {
                let i = *left;
                let a_n = self.project.clips[i].media().len() as i64;
                let a_in = self.project.clips[i].in_off as i64;
                let b_out = self.project.clips[i + 1].out_off as i64;
                let d = ((p.x - start_x) / p0.max(1e-6)).round() as i64;
                let d = d
                    .clamp(a_in - *start_out as i64, a_n - 1 - *start_out as i64)
                    .clamp(-(*start_in as i64), b_out - *start_in as i64);
                self.project.clips[i].out_off = (*start_out as i64 + d) as usize;
                self.project.clips[i + 1].in_off = (*start_in as i64 + d) as usize;
                self.project.clips[i + 1].start = (*b_start as i64 + d) as usize;
                roll_overlay = Some(i);
                edited_clip = Some(i);
            }
            (Some(TlDrag::Slip { clip, start_in, start_out, start_x, ppf: p0 }), Some(p))
                if response.dragged() =>
            {
                let c = &mut self.project.clips[*clip];
                let n = c.media().len() as i64;
                let d = ((p.x - start_x) / p0.max(1e-6)).round() as i64;
                let d = d.clamp(*start_out as i64 - (n - 1), *start_in as i64);
                c.in_off = (*start_in as i64 - d) as usize;
                c.out_off = (*start_out as i64 - d) as usize;
                slip_overlay = Some(*clip);
                edited_clip = Some(*clip);
            }
            (Some(TlDrag::Move { clip, grab }), Some(p)) if response.dragged() => {
                let desired = ((p.x - origin_x) / ppf) as i64 - grab;
                let target =
                    model::resolve_start(&self.project.clips, self.project.clips[*clip].len(), desired, Some(*clip));
                move_ghost = Some((*clip, target));
            }
            (Some(TlDrag::HBar), Some(p)) if response.dragged() => {
                let vis = rect.width() / ppf;
                self.tl_scroll = (((p.x - rect.left()) / rect.width().max(1.0)) * total as f32 - vis * 0.5)
                    .clamp(0.0, (total as f32 - vis).max(0.0));
            }
            (Some(TlDrag::Scrub), _) if response.dragged() => {
                if let Some(pos) = response.interact_pointer_pos() {
                    let f = ((pos.x - origin_x) / ppf).clamp(0.0, total as f32 - 1.0);
                    self.set_playhead(f.floor() as f64);
                    *playing = false;
                }
            }
            _ => {}
        }
        // A trim/roll/slip changes which plate frame sits under the
        // playhead: re-seek so the picture follows the edit.
        if edited_clip.is_some() {
            self.seek_request = Some(self.playhead);
        }
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                match hit(pos) {
                    TlHit::Body(i) | TlHit::Trim(i, _) => self.selected = Some(i),
                    TlHit::Ruler | TlHit::Empty => {
                        let f = ((pos.x - origin_x) / ppf).clamp(0.0, total as f32 - 1.0);
                        self.set_playhead(f.floor() as f64);
                        *playing = false;
                    }
                    TlHit::HBar => {
                        let vis = rect.width() / ppf;
                        self.tl_scroll = (((pos.x - rect.left()) / rect.width().max(1.0)) * total as f32
                            - vis * 0.5)
                            .clamp(0.0, (total as f32 - vis).max(0.0));
                    }
                    TlHit::Roll(_) | TlHit::Playhead => {}
                }
            }
        }
        if response.drag_stopped() {
            if let (Some(TlDrag::Move { clip, grab }), Some(p)) = (&self.tl_drag, pointer) {
                let i = *clip;
                let desired = ((p.x - origin_x) / ppf) as i64 - grab;
                let target =
                    model::resolve_start(&self.project.clips, self.project.clips[i].len(), desired, Some(i));
                let uid = self.project.clips[i].uid;
                self.project.clips[i].start = target;
                self.selected = self.project.sort_timeline(uid);
                self.seek_request = Some(self.playhead);
            }
            self.tl_drag = None;
        }

        if let (Some(p), Some(pos)) = (release_payload, pointer) {
            let desired = ((pos.x - origin_x) / ppf).round() as i64;
            self.place_from_bin(p.0, desired);
        }

        let total = self.project.total_len().max(1);
        if self.playhead >= total as f64 {
            self.playhead = if *playing { self.playhead % total as f64 } else { total as f64 - 1.0 };
        }
        self.playhead = self.playhead.max(0.0);
        let max_scroll = (total as f32 - rect.width() / ppf).max(0.0);
        self.tl_scroll = self.tl_scroll.clamp(0.0, max_scroll);
        let origin_x = rect.left() - self.tl_scroll * ppf;

        painter.rect_filled(ruler, 2.0, theme::timeline_ruler());
        let fps = self.project.fps.max(1.0) as usize;
        let mut f = (self.tl_scroll as usize / fps) * fps;
        while f < total {
            let tx = origin_x + f as f32 * ppf;
            if tx > rect.right() {
                break;
            }
            painter.line_segment(
                [egui::pos2(tx, ruler.bottom() - 4.0), egui::pos2(tx, ruler.bottom())],
                egui::Stroke::new(1.0, theme::lift(theme::border(), 30)),
            );
            f += fps;
        }

        let palette = [
            theme::timeline_clip_1(),
            theme::timeline_clip_2(),
            theme::timeline_clip_3(),
            theme::timeline_clip_4(),
            theme::timeline_clip_5(),
            theme::timeline_clip_6(),
        ];

        let states: Vec<Option<CacheState>> =
            self.project.clips.iter().map(|c| self.cache_state(c.uid)).collect();

        let viz = trim_overlay.or(slip_overlay).or(self.selected.filter(|s| *s < self.project.clips.len()));
        if let Some(vi) = viz {
            let c = &self.project.clips[vi];
            let nm = c.media().len();
            let head = c.in_off;
            let tail = nm - 1 - c.out_off;
            let bx = origin_x + c.start as f32 * ppf;
            let bw = c.len() as f32 * ppf;
            let (top, bot) = (blocks_top + 2.0, hbar_top - 6.0);
            let fill = theme::timeline_handles().gamma_multiply_u8(14);
            let line = theme::timeline_handles().gamma_multiply_u8(60);
            if head > 0 {
                let r = egui::Rect::from_min_max(egui::pos2(bx - head as f32 * ppf, top), egui::pos2(bx, bot));
                painter.rect_filled(r, 2.0, fill);
                painter.rect_stroke(r, 2.0, egui::Stroke::new(1.0, line), egui::StrokeKind::Inside);
                if r.width() > 26.0 {
                    painter.text(
                        r.right_center() + egui::vec2(-4.0, 0.0),
                        egui::Align2::RIGHT_CENTER,
                        format!("{head}f"),
                        egui::FontId::monospace(10.0),
                        line,
                    );
                }
            }
            if tail > 0 {
                let r = egui::Rect::from_min_max(
                    egui::pos2(bx + bw, top),
                    egui::pos2(bx + bw + tail as f32 * ppf, bot),
                );
                painter.rect_filled(r, 2.0, fill);
                painter.rect_stroke(r, 2.0, egui::Stroke::new(1.0, line), egui::StrokeKind::Inside);
                if r.width() > 26.0 {
                    painter.text(
                        r.left_center() + egui::vec2(4.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        format!("{tail}f"),
                        egui::FontId::monospace(10.0),
                        line,
                    );
                }
            }
        }

        for (i, clip) in self.project.clips.iter().enumerate() {
            let x = origin_x + clip.start as f32 * ppf;
            let block = block_of(x, clip.len());
            let mut color = palette[i % palette.len()];
            if self.selected == Some(i) {
                color = color.gamma_multiply(1.5);
            }
            if matches!(move_ghost, Some((c, _)) if c == i) {
                color = color.gamma_multiply(0.5);
            }
            painter.rect_filled(block, 3.0, color);
            if self.selected == Some(i) {
                painter.rect_stroke(block, 3.0, egui::Stroke::new(1.5, theme::selection()), egui::StrokeKind::Outside);
            }
            if Some(clip.uid) == self.current_clip {
                painter.rect_stroke(
                    block.shrink(1.0),
                    3.0,
                    egui::Stroke::new(1.0, theme::timeline_playhead()),
                    egui::StrokeKind::Inside,
                );
            }
            if block.width() >= 18.0 {
                let n = clip.media().len();
                for (hx, has_handle) in
                    [(block.left() + 1.5, clip.in_off > 0), (block.right() - 1.5, clip.out_off + 1 < n)]
                {
                    let alpha = if has_handle { 160 } else { 45 };
                    painter.line_segment(
                        [egui::pos2(hx, block.top() + 3.0), egui::pos2(hx, block.bottom() - 3.0)],
                        egui::Stroke::new(2.0, theme::timeline_handles().gamma_multiply_u8(alpha)),
                    );
                }
            }
            let newer = clip.latest() != clip.current;
            let label = format!("{} {}{}", clip.name, clip.version().label, if newer { "  ⬆" } else { "" });
            painter.text(
                block.left_center() + egui::vec2(6.0, 0.0),
                egui::Align2::LEFT_CENTER,
                label,
                egui::FontId::proportional(12.0),
                theme::text_hot(),
            );
            if trim_overlay == Some(i) || slip_overlay == Some(i) {
                let verb = if slip_overlay == Some(i) { "slip  " } else { "" };
                painter.text(
                    block.center_top() + egui::vec2(0.0, -2.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("{verb}in {} · out {} · {}f", clip.in_off, clip.out_off, clip.len()),
                    egui::FontId::monospace(11.0),
                    theme::timeline_active(),
                );
            }
            if roll_overlay == Some(i) {
                let b = &self.project.clips[i + 1];
                painter.text(
                    egui::pos2(block.right(), block.top() - 2.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("roll  out {} | in {}", clip.out_off, b.in_off),
                    egui::FontId::monospace(11.0),
                    theme::timeline_active(),
                );
            }
            let strip = egui::Rect::from_min_max(egui::pos2(block.left(), block.bottom() - 4.0), block.right_bottom());
            match &states[i] {
                Some(CacheState::Ready(_)) => {
                    painter.rect_filled(strip, 1.0, theme::ok());
                }
                Some(CacheState::Encoding(p)) => {
                    let mut r = strip;
                    r.set_width(strip.width() * p);
                    painter.rect_filled(r, 1.0, theme::warn());
                    ui.ctx().request_repaint_after(web_time::Duration::from_millis(200));
                }
                Some(CacheState::Failed(_)) => {
                    painter.rect_filled(strip, 1.0, theme::error());
                }
                None => {}
            }
        }

        let ghost = |start: usize, len: usize, color: egui::Color32| {
            let r = egui::Rect::from_min_max(
                egui::pos2(origin_x + start as f32 * ppf, blocks_top),
                egui::pos2(origin_x + (start + len) as f32 * ppf, hbar_top - 4.0),
            );
            painter.rect_filled(r, 3.0, color.gamma_multiply(0.25));
            painter.rect_stroke(r, 3.0, egui::Stroke::new(1.5, color), egui::StrokeKind::Inside);
        };
        if let (Some(pl), Some(pos)) = (&hover_payload, pointer) {
            if let Some(bc) = self.project.bin.iter().find(|c| c.uid == pl.0) {
                let desired = ((pos.x - origin_x) / ppf).round() as i64;
                let start = model::resolve_start(&self.project.clips, bc.len(), desired, None);
                ghost(start, bc.len(), theme::timeline_active());
            }
        }
        if let Some((ci, target)) = move_ghost {
            ghost(target, self.project.clips[ci].len(), theme::timeline_move());
        }

        let bar = egui::Rect::from_min_max(egui::pos2(rect.left(), hbar_top + 1.0), egui::pos2(rect.right(), rect.bottom() - 1.0));
        painter.rect_filled(bar, 3.0, theme::timeline_ruler());
        let vis = rect.width() / ppf;
        if (total as f32) > vis {
            let frac0 = (self.tl_scroll / total as f32).clamp(0.0, 1.0);
            let frac1 = ((self.tl_scroll + vis) / total as f32).clamp(0.0, 1.0);
            let thumb = egui::Rect::from_min_max(
                egui::pos2(rect.left() + frac0 * rect.width(), bar.top()),
                egui::pos2(rect.left() + frac1 * rect.width(), bar.bottom()),
            );
            painter.rect_filled(thumb, 3.0, theme::lift(theme::button(), 20));
        }

        let px = origin_x + self.playhead as f32 * ppf;
        let ph_color = theme::timeline_playhead();
        painter.line_segment([egui::pos2(px, rect.top()), egui::pos2(px, hbar_top)], egui::Stroke::new(1.5, ph_color));
        painter.add(egui::Shape::convex_polygon(
            vec![egui::pos2(px - 5.0, rect.top()), egui::pos2(px + 5.0, rect.top()), egui::pos2(px, rect.top() + 10.0)],
            ph_color,
            egui::Stroke::NONE,
        ));

        self.transport_row(ui, playing, ppf, rect.width());
    }

    fn transport_row(&mut self, ui: &mut egui::Ui, playing: &mut bool, ppf: f32, view_w: f32) {
        let total = self.project.total_len();
        ui.horizontal(|ui| {
            let icon = if *playing { "⏸" } else { "▶" };
            if ui.button(icon).clicked() {
                *playing = !*playing;
            }
            if ui
                .add_enabled(!self.history.is_empty(), egui::Button::new("↺"))
                .on_hover_text("undo the last timeline or bin edit (⌘Z over the bin or timeline)")
                .clicked()
            {
                self.undo();
            }
            if ui
                .add_enabled(!self.future.is_empty(), egui::Button::new("↻"))
                .on_hover_text("redo (⌘⇧Z over the bin or timeline)")
                .clicked()
            {
                self.redo();
            }
            if ui
                .button("enter wobble ⚙")
                .on_hover_text("open the comp editor on the shot under the playhead — same instance, its own window")
                .clicked()
            {
                self.enter_wobble = true;
            }
            if ui.small_button("fit").on_hover_text("zoom to fit the whole edit (F)").clicked() {
                self.tl_zoom = None;
                self.tl_scroll = 0.0;
            }
            if ui.small_button("−").on_hover_text("zoom out (-)").clicked() {
                self.zoom_anchored(ppf / 1.5, view_w);
            }
            if ui.small_button("＋").on_hover_text("zoom in (=)").clicked() {
                self.zoom_anchored(ppf * 1.5, view_w);
            }
            let mut lg = ppf.log2();
            let slider = ui.add(
                egui::Slider::new(&mut lg, (-5.6)..=7.6)
                    .show_value(false)
                    .handle_shape(egui::style::HandleShape::Rect { aspect_ratio: 0.5 }),
            );
            if slider.changed() {
                self.zoom_anchored(2f32.powf(lg), view_w);
            }
            slider.on_hover_text("zoom");
            let total_f = total.max(1);
            let global = (self.playhead.floor() as usize).min(total_f - 1);
            match self.project.locate(global) {
                Some((ci, local)) => {
                    let clip = &self.project.clips[ci];
                    ui.monospace(format!(
                        "{global:05} / {total:05}   {} {}  f{}",
                        clip.name,
                        clip.version().label,
                        clip.in_off + local
                    ));
                }
                None => {
                    ui.monospace(format!("{global:05} / {total:05}   — gap —"));
                }
            }
            ui.separator();
            ui.label("fps");
            ui.add(egui::DragValue::new(&mut self.project.fps).update_while_editing(false).range(1.0..=120.0).speed(0.1));
            ui.separator();
            if ui.button("open…").clicked() {
                if let Some(f) = crate::filedialog::FileDialog::new().add_filter("wobble edit", &["wedit"]).pick_file() {
                    self.open_request = Some(f);
                }
            }
            let dirty = self.is_dirty();
            if ui
                .button(if dirty { "save*" } else { "save" })
                .on_hover_text(if dirty { "unsaved changes" } else { "saved" })
                .clicked()
            {
                let target = self.project_path.clone().or_else(|| {
                    crate::filedialog::FileDialog::new()
                        .add_filter("wobble edit", &["wedit"])
                        .set_file_name("edit.wedit")
                        .save_file()
                });
                if let Some(t) = target {
                    self.save_request = Some(t);
                }
            }
            if ui.button("export true frames…").clicked() && total > 0 {
                if let Some(dir) = crate::filedialog::FileDialog::new().pick_folder() {
                    super::export::export_copy(&self.project, dir, self.export_status.clone());
                }
            }
            if let Some(msg) = self.export_status.0.lock().unwrap().clone() {
                ui.label(egui::RichText::new(msg).weak());
            }
            if ui.button("clear proxies").on_hover_text("delete every ProRes proxy on disk").clicked() {
                let n = self.cache.clear();
                self.cache_keys.clear();
                self.error = Some(format!("removed {n} proxy files"));
            }
            ui.label(
                egui::RichText::new("space play · ruler scrubs · edges trim · cuts roll · drag moves · ⌥drag slips")
                    .weak(),
            );
            if let Some(e) = &self.error {
                ui.separator();
                let e = e.clone();
                ui.colored_label(theme::error(), &e);
                if ui.small_button("×").clicked() {
                    self.error = None;
                }
            }
        });
    }
}

#[cfg(test)]
mod dnd_tests {
    use super::super::EditState;
    use crate::edit::model::Clip;

    fn frame(ctx: &egui::Context, e: &mut EditState, t: f64, events: Vec<egui::Event>) {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 400.0))),
            time: Some(t),
            events,
            ..Default::default()
        };
        let mut playing = false;
        let _ = ctx.run_ui(raw, |ui| {
            // The edit's arrangement: bin on the left, timeline in the rest.
            egui::Panel::left(egui::Id::new("bin")).exact_size(320.0).show(ui, |ui| e.bin_ui(ui));
            egui::CentralPanel::default().show(ui, |ui| e.timeline_ui(ui, &mut playing));
        });
    }

    fn press(pos: egui::Pos2, pressed: bool) -> egui::Event {
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        }
    }

    /// A bin row's ≡ dragged onto the timeline places the clip — through the
    /// real widgets and real synthetic pointer events, so an occluder or a
    /// broken drop target fails this rather than the user.
    #[test]
    fn dragging_a_bin_handle_onto_the_timeline_places_the_clip() {
        let ctx = egui::Context::default();
        let mut e = EditState::for_test();
        e.project.bin.push(Clip::for_test("plate", 100));
        let uid = e.project.bin[0].uid;

        // Lay out once so the handle and the strip have rectangles.
        frame(&ctx, &mut e, 0.0, vec![]);
        frame(&ctx, &mut e, 0.05, vec![]);
        let handle = *e.handle_rects.get(&uid).expect("the row drew its drag handle");
        let strip = e.tl_rect.expect("the timeline drew its strip");
        let from = handle.center();
        let to = strip.center();
        assert!(!strip.contains(from), "handle must be outside the strip for this to mean anything");

        frame(&ctx, &mut e, 0.10, vec![egui::Event::PointerMoved(from)]);
        frame(&ctx, &mut e, 0.15, vec![press(from, true)]);
        frame(&ctx, &mut e, 0.20, vec![egui::Event::PointerMoved(from + egui::vec2(12.0, 0.0))]);
        frame(&ctx, &mut e, 0.25, vec![egui::Event::PointerMoved(from + egui::vec2(40.0, 20.0))]);
        frame(&ctx, &mut e, 0.30, vec![egui::Event::PointerMoved(to)]);
        frame(&ctx, &mut e, 0.35, vec![egui::Event::PointerMoved(to)]);
        assert!(
            e.project.clips.is_empty(),
            "nothing lands before the button is released"
        );
        frame(&ctx, &mut e, 0.40, vec![press(to, false)]);
        frame(&ctx, &mut e, 0.45, vec![]);

        assert_eq!(e.project.clips.len(), 1, "the drop placed the clip");
        assert_eq!(e.project.clips[0].master, Some(uid), "it is an instance of the bin clip");
        assert_eq!(e.new_clips, vec![uid], "and its master needs a shot minted");
    }

    /// The same drop while the SOURCE viewer is open — the strip sits above
    /// the timeline then, exactly the arrangement the user drags into.
    #[test]
    fn dragging_onto_the_timeline_works_in_source_mode_too() {
        let ctx = egui::Context::default();
        let mut e = EditState::for_test();
        e.project.bin.push(Clip::for_test("plate", 100));
        let uid = e.project.bin[0].uid;
        e.source = Some(uid);
        let mut playing = false;
        let mut run = |e: &mut EditState, t: f64, events: Vec<egui::Event>| {
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 400.0))),
                time: Some(t),
                events,
                ..Default::default()
            };
            let _ = ctx.run_ui(raw, |ui| {
                egui::Panel::left(egui::Id::new("bin")).exact_size(320.0).show(ui, |ui| e.bin_ui(ui));
                egui::Panel::bottom(egui::Id::new("edit-bottom")).exact_size(210.0).show(ui, |ui| {
                    e.source_ui(ui, 10, &mut playing);
                    e.timeline_ui(ui, &mut playing);
                });
            });
        };
        run(&mut e, 0.0, vec![]);
        run(&mut e, 0.05, vec![]);
        let from = e.handle_rects[&uid].center();
        let to = e.tl_rect.unwrap().center();
        run(&mut e, 0.10, vec![egui::Event::PointerMoved(from)]);
        run(&mut e, 0.15, vec![press(from, true)]);
        run(&mut e, 0.20, vec![egui::Event::PointerMoved(from + egui::vec2(12.0, 0.0))]);
        run(&mut e, 0.25, vec![egui::Event::PointerMoved(to)]);
        run(&mut e, 0.30, vec![egui::Event::PointerMoved(to)]);
        run(&mut e, 0.35, vec![press(to, false)]);
        run(&mut e, 0.40, vec![]);
        assert_eq!(e.project.clips.len(), 1, "the drop placed the clip with the source strip open");
        assert_eq!(e.source, Some(uid), "and dropping did not itself leave the source viewer");
    }

    /// The row's buttons must actually be reachable — the bug this guards
    /// against was a row-wide click target sitting on top of them.
    #[test]
    fn the_view_button_is_clickable() {
        let ctx = egui::Context::default();
        let mut e = EditState::for_test();
        e.project.bin.push(Clip::for_test("plate", 100));
        let uid = e.project.bin[0].uid;
        frame(&ctx, &mut e, 0.0, vec![]);
        frame(&ctx, &mut e, 0.05, vec![]);
        // Find the button by asking egui where it drew it.
        let handle = *e.handle_rects.get(&uid).unwrap();
        // The "▶ view" button sits two rows under the handle row; probe a
        // small grid below the handle and click the first spot that flips
        // the source request.
        let mut hit = None;
        let mut t = 1.0f64;
        'outer: for dy in [24.0f32, 30.0, 36.0, 42.0, 48.0] {
            for dx in [0.0f32, 10.0, 20.0, 30.0] {
                let p = handle.left_center() + egui::vec2(dx, dy);
                t += 1.0;
                frame(&ctx, &mut e, t, vec![egui::Event::PointerMoved(p)]);
                frame(&ctx, &mut e, t + 0.1, vec![press(p, true)]);
                frame(&ctx, &mut e, t + 0.2, vec![press(p, false)]);
                frame(&ctx, &mut e, t + 0.3, vec![]);
                if e.source_request.is_some() {
                    hit = Some(p);
                    break 'outer;
                }
            }
        }
        assert!(hit.is_some(), "no click below the handle reached the ▶ view button");
        assert_eq!(e.source_request, Some(Some(uid)));
    }
}
