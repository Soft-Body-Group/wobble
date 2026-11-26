//! Titles and subtitles in the editor: making text clips, the text window
//! (the clip's words, and its STYLE: font, size, colour, alignment, line
//! spacing, outline, shadow, box, position), drawing text over the record
//! picture, moving a style by dragging its text there (or cmd+arrows), and
//! SRT in and out.
//!
//! - cmd+shift+T (or sequence > new text, or the bin's "T text" dragged onto
//!   a video track) makes a 3-second text clip at the playhead, on the track
//!   above the targeted one where there is room, in the current style.
//! - Enter on a selected text clip, or a double-click on it, opens the text
//!   window on it. Typing changes that clip only; the style section changes
//!   every clip using the style (that is the point: retype a duplicated
//!   subtitle and it stays put).
//! - In the record viewer, drag a text to move its STYLE (every clip of the
//!   style moves with it); cmd+arrows nudge the selected text clip's style.

use crate::editor::Editor;
use crate::timelineui::Selection;
use egui::{pos2, vec2, Color32, Rect};
use model::{Align, TextStyle, VAlign, PLACES};
use std::path::{Path, PathBuf};

/// Text clips start this long.
pub const TEXT_SECS: f64 = 3.0;
/// The viewer draws text made at most this tall (the render makes its own
/// at the output's size).
const PREVIEW_H: u32 = 1080;

/// The viewer's text textures and what was drawn where.
#[derive(Default)]
pub struct TextView {
    tex: std::collections::HashMap<u64, Option<(egui::TextureHandle, crate::text::Bitmap)>>,
    /// Last frame's text blocks on screen: (track, start, style, rect).
    pub boxes: Vec<(usize, i64, usize, Rect)>,
    /// A drag moving a style: (style, its x, y at the press).
    drag: Option<(usize, f32, f32)>,
    /// The text window: the clip it edits (track, start), and focus once.
    pub open: Option<(usize, i64)>,
    pub focus: bool,
    /// The style new text clips take.
    pub style: usize,
    /// Font search in the picker.
    pub font_find: String,
}

impl Editor {
    /// The selected clip, when it is a text clip: (track, index).
    pub(crate) fn selected_text(&self) -> Option<(usize, usize)> {
        let (t, i) = match &self.rec.sel {
            Selection::Clip { track, start } => (*track, self.rec.seq.tracks[*track].clips.iter().position(|c| c.start == *start)?),
            _ => return None,
        };
        self.rec.seq.tracks[t].clips[i].text.is_some().then_some((t, i))
    }

    /// A new text clip at `at` (the playhead when `None`): on `track`, or
    /// the lowest video track above the targeted one that is free there.
    /// Selected, and the text window opened on it.
    pub(crate) fn new_text(&mut self, track: Option<usize>, at: Option<i64>, text: &str) -> Option<(usize, i64)> {
        let at = at.unwrap_or(self.rec.transport.frame() as i64);
        let len = (TEXT_SECS * self.rec.seq.fps.as_f64()).round() as i64;
        let seq = &self.rec.seq;
        let free = |t: usize| seq.tracks[t].clips.iter().all(|c| c.end() <= at || c.start >= at + len);
        let track = track.unwrap_or_else(|| {
            let base = self.rec.targets.video.unwrap_or(model::V1);
            model::VIDEO_TRACKS.into_iter().filter(|&t| t > base).find(|&t| free(t)).unwrap_or(model::V3.max(base))
        });
        self.rec.seq.ensure_styles();
        let style = self.textv.style.min(self.rec.seq.styles.len() - 1);
        self.rec.seq.add_text(track, at, len, text, style)?;
        let start = at.max(0);
        self.rec.sel = Selection::Clip { track, start };
        self.textv.open = Some((track, start));
        self.textv.focus = true;
        self.changed(false);
        self.say(format!("text on {} (Enter edits, double-click too)", self.rec.seq.tracks[track].name));
        Some((track, start))
    }

    /// Enter / double-click: the text window on this clip.
    pub(crate) fn open_text(&mut self, track: usize, start: i64) {
        self.textv.open = Some((track, start));
        self.textv.focus = true;
    }

    /// cmd+arrows on a text clip: its style moves (fractions of the frame).
    pub(crate) fn nudge_text(&mut self, dx: f32, dy: f32) -> bool {
        let Some((t, i)) = self.selected_text() else { return false };
        let s = self.rec.seq.tracks[t].clips[i].text.as_ref().map_or(0, |x| x.style);
        if let Some(st) = self.rec.seq.styles.get_mut(s) {
            st.x = (st.x + dx).clamp(-0.5, 1.5);
            st.y = (st.y + dy).clamp(-0.5, 1.5);
            let m = format!("style \"{}\" at {:.0}% {:.0}%", st.name, st.x * 100.0, st.y * 100.0);
            self.changed(true);
            self.say(m);
        }
        true
    }

    /// The text over the record picture (drawn into `raster_rect`), and a
    /// drag on it moving its style. `hand` is the viewer's response.
    pub(crate) fn text_overlay(&mut self, ui: &egui::Ui, raster_rect: Rect, raster: (u32, u32), t: i64, hand: &egui::Response, navigating: bool) {
        let s = (PREVIEW_H as f32 / raster.1.max(1) as f32).min(1.0);
        let size = (((raster.0 as f32 * s).round() as u32).max(8), ((raster.1 as f32 * s).round() as u32).max(8));
        let painter = ui.painter_at(raster_rect);
        let mut boxes = Vec::new();
        // What draws now, and (warmed, not drawn) what draws a second on.
        let ahead = t + self.rec.seq.fps.as_f64().round() as i64;
        let mut todo: Vec<(model::TextLayer, bool)> = self.rec.seq.text_at(t).into_iter().map(|l| (l, true)).collect();
        if !self.rec.transport.playing() || self.textv.tex.len() < 200 {
            todo.extend(self.rec.seq.text_at(ahead).into_iter().map(|l| (l, false)));
        }
        if self.textv.tex.len() > 256 {
            self.textv.tex.clear();
        }
        for (l, draw) in todo {
            let k = crate::text::layer_key(&l, size);
            let entry = self.textv.tex.entry(k).or_insert_with(|| {
                crate::text::rasterize_layer(&l, size).map(|b| {
                    let img = egui::ColorImage::from_rgba_premultiplied([b.w as usize, b.h as usize], &b.rgba);
                    let tex = ui.ctx().load_texture(format!("text-{k}"), img, egui::TextureOptions::LINEAR);
                    (tex, crate::text::Bitmap { rgba: Vec::new(), ..b })
                })
            });
            if !draw {
                continue;
            }
            let Some((tex, b)) = entry else { continue };
            let sx = raster_rect.width() / size.0 as f32;
            let sy = raster_rect.height() / size.1 as f32;
            let r = Rect::from_min_size(raster_rect.min + vec2(b.x as f32 * sx, b.y as f32 * sy), vec2(b.w as f32 * sx, b.h as f32 * sy));
            let o = (l.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
            painter.image(tex.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::from_rgba_premultiplied(o, o, o, o));
            boxes.push(r);
        }
        // Which clips those blocks are (same order: tracks bottom up).
        let mut owners = Vec::new();
        for &ti in model::VIDEO_TRACKS.iter() {
            if let Some(c) = self.rec.seq.tracks[ti].at(t) {
                if let Some(x) = &c.text {
                    owners.push((ti, c.start, x.style));
                }
            }
        }
        self.textv.boxes = owners.into_iter().zip(boxes).map(|((a, b, c), r)| (a, b, c, r)).collect();
        // Selected text: a thin frame round it.
        if let Some((st, ss)) = self.selected_text().map(|(t, i)| (t, self.rec.seq.tracks[t].clips[i].start)) {
            if let Some(r) = self.textv.boxes.iter().find(|b| b.0 == st && b.1 == ss).map(|b| b.3) {
                painter.rect_stroke(r, 2.0, egui::Stroke::new(1.0, Color32::from_rgb(255, 210, 80)), egui::StrokeKind::Outside);
            }
        }
        // Drag a text: its style moves.
        let plain = !ui.input(|i| i.modifiers.alt || i.modifiers.command);
        if !navigating && plain && hand.drag_started_by(egui::PointerButton::Primary) {
            let q = ui.input(|i| i.pointer.press_origin()).unwrap_or_default();
            if let Some(&(tr, start, style, _)) = self.textv.boxes.iter().rev().find(|b| b.3.contains(q)) {
                if let Some(s) = self.rec.seq.styles.get(style) {
                    self.textv.drag = Some((style, s.x, s.y));
                    self.rec.sel = Selection::Clip { track: tr, start };
                }
            }
        }
        if let Some((style, x0, y0)) = self.textv.drag {
            if hand.dragged_by(egui::PointerButton::Primary) {
                let d = ui.input(|i| i.pointer.interact_pos().zip(i.pointer.press_origin()).map(|(a, b)| a - b)).unwrap_or_default();
                if let Some(s) = self.rec.seq.styles.get_mut(style) {
                    s.x = x0 + d.x / raster_rect.width().max(1.0);
                    s.y = y0 + d.y / raster_rect.height().max(1.0);
                }
                self.changed(true);
            }
            if hand.drag_stopped() || !ui.input(|i| i.pointer.primary_down()) {
                self.textv.drag = None;
                self.changed(false);
            }
        }
    }

    /// The text window: the clip's words and its style.
    pub(crate) fn text_window(&mut self, ctx: &egui::Context) {
        let Some((track, start)) = self.textv.open else { return };
        let Some(idx) = self.rec.seq.tracks.get(track).and_then(|t| t.clips.iter().position(|c| c.start == start && c.text.is_some())) else {
            self.textv.open = None;
            return;
        };
        let mut open = true;
        let mut changed = false;
        let mut settled = false;
        let mut new_style = false;
        let mut pick_font = false;
        let fonts = crate::text::fonts();
        let focus = std::mem::take(&mut self.textv.focus);
        let seq = &mut self.rec.seq;
        seq.ensure_styles();
        let n_styles = seq.styles.len();
        egui::Window::new("text").open(&mut open).collapsible(false).resizable(true).default_width(360.0).show(ctx, |ui| {
            let clip = &mut seq.tracks[track].clips[idx];
            let ct = clip.text.as_mut().expect("checked");
            ui.label(format!("{} at {}  (this clip's words; Esc closes)", model::TRACKS[track].0, start));
            let r = ui.add(egui::TextEdit::multiline(&mut ct.text).desired_rows(3).desired_width(f32::INFINITY));
            if focus {
                r.request_focus();
            }
            changed |= r.changed();
            settled |= r.lost_focus();
            let mut si = ct.style;
            ui.horizontal(|ui| {
                ui.label("style");
                egui::ComboBox::from_id_salt("text-style").selected_text(seq.styles[si.min(n_styles - 1)].name.clone()).show_ui(ui, |ui| {
                    for (k, s) in seq.styles.iter().enumerate() {
                        ui.selectable_value(&mut si, k, s.name.clone());
                    }
                });
                new_style = ui.button("new style from this").clicked();
            });
            let clip = &mut seq.tracks[track].clips[idx];
            let ct = clip.text.as_mut().expect("checked");
            if si != ct.style {
                ct.style = si;
                changed = true;
                settled = true;
            }
            ui.separator();
            let users = seq.tracks.iter().flat_map(|t| &t.clips).filter(|c| c.text.as_ref().is_some_and(|x| x.style == si)).count();
            ui.weak(format!("the style: changes every clip using it ({users})"));
            let s = &mut seq.styles[si];
            let before = s.clone();
            ui.horizontal(|ui| {
                ui.label("name");
                ui.text_edit_singleline(&mut s.name);
            });
            ui.horizontal(|ui| {
                ui.label("font");
                let cur = s.font.as_ref().and_then(|p| p.file_stem()).map_or("built-in".to_string(), |n| n.to_string_lossy().into_owned());
                egui::ComboBox::from_id_salt("text-font").selected_text(cur).height(320.0).show_ui(ui, |ui| {
                    ui.text_edit_singleline(&mut self.textv.font_find);
                    let find = self.textv.font_find.to_lowercase();
                    if ui.selectable_label(s.font.is_none(), "built-in").clicked() {
                        s.font = None;
                    }
                    for (name, path) in fonts.iter().filter(|(n, _)| find.is_empty() || n.to_lowercase().contains(&find)).take(400) {
                        if ui.selectable_label(s.font.as_ref() == Some(path), name).clicked() {
                            s.font = Some(path.clone());
                            s.font_index = 0;
                        }
                    }
                });
                pick_font = ui.button("file…").clicked();
                ui.add(egui::DragValue::new(&mut s.font_index).range(0..=64).prefix("face "));
            });
            ui.horizontal(|ui| {
                ui.label("size");
                ui.add(egui::Slider::new(&mut s.size, 0.01..=0.4).custom_formatter(|v, _| format!("{:.1}% of height", v * 100.0)));
                ui.color_edit_button_srgba_unmultiplied(&mut s.color);
            });
            ui.horizontal(|ui| {
                ui.label("align");
                ui.selectable_value(&mut s.align, Align::Left, "left");
                ui.selectable_value(&mut s.align, Align::Center, "centre");
                ui.selectable_value(&mut s.align, Align::Right, "right");
                ui.label("lines");
                ui.add(egui::DragValue::new(&mut s.line_spacing).range(0.5..=3.0).speed(0.01));
            });
            ui.horizontal(|ui| {
                ui.label("outline");
                ui.add(egui::Slider::new(&mut s.outline, 0.0..=0.2));
                ui.color_edit_button_srgba_unmultiplied(&mut s.outline_color);
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut s.shadow, "shadow");
                ui.color_edit_button_srgba_unmultiplied(&mut s.shadow_color);
                ui.checkbox(&mut s.boxed, "box");
                ui.color_edit_button_srgba_unmultiplied(&mut s.box_color);
                ui.add(egui::DragValue::new(&mut s.box_pad).range(0.0..=2.0).speed(0.01).prefix("pad "));
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("place");
                for (k, p) in PLACES.iter().enumerate() {
                    if ui.small_button(p.0).clicked() {
                        s.place(k);
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("at");
                ui.add(egui::DragValue::new(&mut s.x).speed(0.002).range(-0.5..=1.5).custom_formatter(|v, _| format!("x {:.1}%", v * 100.0)));
                ui.add(egui::DragValue::new(&mut s.y).speed(0.002).range(-0.5..=1.5).custom_formatter(|v, _| format!("y {:.1}%", v * 100.0)));
                ui.selectable_value(&mut s.valign, VAlign::Top, "top");
                ui.selectable_value(&mut s.valign, VAlign::Middle, "middle");
                ui.selectable_value(&mut s.valign, VAlign::Bottom, "bottom");
            });
            ui.weak("drag the text in the record viewer, or cmd+arrows, to move the style");
            if *s != before {
                changed = true;
                settled |= !ui.input(|i| i.pointer.any_down());
            }
        });
        if pick_font {
            if let Some(p) = imfram::filedialog::FileDialog::new().add_filter("fonts", &["ttf", "otf", "ttc"]).set_directory("/Library/Fonts").pick_file() {
                let si = self.rec.seq.tracks[track].clips[idx].text.as_ref().map_or(0, |x| x.style);
                if crate::text::loads(&p, 0) {
                    self.rec.seq.styles[si].font = Some(p);
                    self.rec.seq.styles[si].font_index = 0;
                    changed = true;
                    settled = true;
                } else {
                    self.say(format!("{} is not a font Scrub can read", p.display()));
                }
            }
        }
        if new_style {
            let si = self.rec.seq.tracks[track].clips[idx].text.as_ref().map_or(0, |x| x.style);
            let mut s: TextStyle = self.rec.seq.styles[si].clone();
            s.name = format!("{} 2", s.name);
            self.rec.seq.styles.push(s);
            let n = self.rec.seq.styles.len() - 1;
            if let Some(ct) = self.rec.seq.tracks[track].clips[idx].text.as_mut() {
                ct.style = n;
            }
            self.textv.style = n;
            changed = true;
            settled = true;
        }
        if changed {
            // The style picked last is the one new text takes.
            self.textv.style = self.rec.seq.tracks[track].clips[idx].text.as_ref().map_or(0, |x| x.style);
            self.changed(!settled);
        }
        if !open {
            self.textv.open = None;
            self.changed(false);
        }
    }

    /// Text clips from an SRT file on `track` (the targeted video track by
    /// default), in the current style. Returns how many.
    pub(crate) fn import_srt(&mut self, path: &Path, track: Option<usize>, style: Option<usize>) -> Result<usize, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        let cues = model::parse_srt(&s);
        if cues.is_empty() {
            return Err(format!("{} has no subtitles Scrub can read", path.display()));
        }
        let track = track.or(self.rec.targets.video).unwrap_or(model::V2);
        let fps = self.rec.seq.fps.as_f64();
        self.rec.seq.ensure_styles();
        let style = style.or(self.rec.seq.style_named("subtitle")).unwrap_or(self.textv.style);
        for (a, b, text) in &cues {
            let (s0, s1) = ((a * fps).round() as i64, (b * fps).round() as i64);
            self.rec.seq.add_text(track, s0, (s1 - s0).max(1), text, style);
        }
        self.changed(false);
        self.say(format!("{} subtitles on {}", cues.len(), self.rec.seq.tracks[track].name));
        Ok(cues.len())
    }

    /// The text clips of `track` (default: the video track with the most)
    /// written as SRT. Returns how many.
    pub(crate) fn export_srt(&mut self, path: &Path, track: Option<usize>) -> Result<usize, String> {
        let track = track.unwrap_or_else(|| model::VIDEO_TRACKS.into_iter().max_by_key(|&t| self.rec.seq.texts_on(t).len()).unwrap_or(model::V2));
        let fps = self.rec.seq.fps.as_f64();
        let cues: Vec<(f64, f64, String)> = self.rec.seq.texts_on(track).into_iter().map(|(a, b, t)| (a as f64 / fps, b as f64 / fps, t)).collect();
        if cues.is_empty() {
            return Err(format!("no text clips on {}", self.rec.seq.tracks[track].name));
        }
        crate::project::write_atomic(path, model::format_srt(&cues).as_bytes()).map_err(|e| format!("writing {}: {e}", path.display()))?;
        self.say(format!("{} subtitles written to {}", cues.len(), path.display()));
        Ok(cues.len())
    }

    /// File menu: pick an SRT to import / a file to export to.
    pub(crate) fn srt_dialog(&mut self, export: bool) {
        let d = imfram::filedialog::FileDialog::new().add_filter("SubRip", &["srt"]);
        let picked: Option<PathBuf> = if export { d.set_file_name(format!("{}.srt", self.project.name())).save_file() } else { d.pick_file() };
        if let Some(p) = picked {
            let r = if export { self.export_srt(&p, None) } else { self.import_srt(&p, None, None) };
            if let Err(e) = r {
                self.say(e);
            }
        }
    }
}
