//! A GenAI node's HISTORY: every take it made, as pictures, and three ways
//! to use one again.
//!
//! - **pick** a thumbnail and the node shows that take (as before);
//! - **settings** puts back the prompt, seed and model that made it
//!   (`genai::history`, the recipe kept beside the result), so a variation
//!   starts from the one you liked rather than from wherever the panel is
//!   now. Nothing runs: it is an edit, and an undo step;
//! - **load** makes a Load node of the take's file beside the node, and
//!   **feed** makes one and wires it into the node's image input, so an
//!   earlier result is the next run's reference.
//!
//! A child module of `genaiui` (it reads the panel's private helpers), in
//! its own file so the history can grow without the panel file doing so.
//!
//! Thumbnails decode off the UI thread, once per file, through the app's
//! own decoders (a take is a png, a jpeg or a movie's first frame). A take
//! is display-referred sRGB, so the code values ARE the thumbnail: no tone
//! curve, unlike a browser thumbnail of a scene-linear plate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use egui::{Color32, Sense, Stroke, Ui};

use super::{small, video_ok, Action, View, NO_VIDEO};
use crate::theme;
use engine::graph::GenAiParams;
use genai::{config, service, Take};

/// Thumbnail height on screen, in points; decoded at twice that.
const THUMB_H: f32 = 38.0;
/// A sheet picture's height on screen, in points; decoded at twice that.
const SHEET_H: f32 = 200.0;

enum Slot {
    Pending,
    Ready(egui::TextureHandle),
    /// Missing, unreadable, or a format this build cannot decode.
    Failed,
}

#[derive(Default)]
struct Thumbs {
    /// By file and by size: a history thumbnail, or a sheet picture.
    map: HashMap<(PathBuf, bool), Slot>,
    done: Vec<((PathBuf, bool), Option<egui::ColorImage>)>,
}

fn thumbs() -> &'static Mutex<Thumbs> {
    static T: OnceLock<Mutex<Thumbs>> = OnceLock::new();
    T.get_or_init(Default::default)
}

/// Decode `path`'s first frame to an sRGB picture `high` points tall.
#[cfg(not(target_arch = "wasm32"))]
fn decode(path: &Path, high: f32) -> Option<egui::ColorImage> {
    use engine::decode::{AnyDecoder, ExrDecoder};
    static DEC: OnceLock<AnyDecoder> = OnceLock::new();
    let dec = DEC.get_or_init(|| AnyDecoder::new(true));
    let info = dec.read_header(path).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    if w == 0 || h == 0 {
        return None;
    }
    let stride = w * 4;
    let mut buf = vec![half::f16::ZERO; stride * h];
    dec.decode_frame_into(path, info, stride, None, 0, &mut buf).ok()?;
    let th = ((high * 2.0) as usize).min(h).max(1);
    let tw = (w * th / h).max(1);
    let mut rgba = vec![255u8; tw * th * 4];
    for y in 0..th {
        let sy = y * h / th;
        for x in 0..tw {
            let s = sy * stride + (x * w / tw) * 4;
            let d = (y * tw + x) * 4;
            for c in 0..3 {
                rgba[d + c] = (buf[s + c].to_f32().clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    Some(egui::ColorImage::from_rgba_unmultiplied([tw, th], &rgba))
}

/// The thumbnail for a take's file (`big`: the sheet's larger picture),
/// asking for it on first sight.
fn thumb(ctx: &egui::Context, path: &Path, big: bool) -> Option<egui::TextureHandle> {
    let key = (path.to_path_buf(), big);
    let mut t = thumbs().lock().unwrap();
    for (p, img) in std::mem::take(&mut t.done) {
        let slot = match img {
            Some(img) => Slot::Ready(ctx.load_texture("genai-take", img, egui::TextureOptions::LINEAR)),
            None => Slot::Failed,
        };
        t.map.insert(p, slot);
    }
    match t.map.get(&key) {
        Some(Slot::Ready(tex)) => return Some(tex.clone()),
        Some(_) => return None,
        None => {}
    }
    // A node keeps every take and a session sees a handful of nodes: this
    // only matters for a very long one, and then starting over is fine.
    if t.map.len() > 512 {
        t.map.clear();
    }
    t.map.insert(key.clone(), Slot::Pending);
    drop(t);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (path, ctx) = (path.to_path_buf(), ctx.clone());
        std::thread::spawn(move || {
            let name = path.to_string_lossy();
            let ok = (!genai::pipeline::is_video_file(&name) || video_ok())
                && !genai::pipeline::is_splat_file(&name)
                && path.exists();
            let img = if ok { decode(&path, if big { SHEET_H } else { THUMB_H }) } else { None };
            drop(name);
            thumbs().lock().unwrap().done.push((key, img));
            ctx.request_repaint();
        });
    }
    #[cfg(target_arch = "wasm32")]
    thumbs().lock().unwrap().done.push((key, None));
    None
}

/// "3 h ago", from a take's unix seconds; empty when it has none.
fn ago(when: u64, now: u64) -> String {
    if when == 0 || now < when {
        return String::new();
    }
    match now - when {
        s if s < 90 => "just now".into(),
        s if s < 90 * 60 => format!("{} min ago", s / 60),
        s if s < 36 * 3600 => format!("{} h ago", s / 3600),
        s => format!("{} d ago", s / 86400),
    }
}

fn now() -> u64 {
    // No clock in a page (and no takes on disk there either).
    #[cfg(target_arch = "wasm32")]
    return 0;
    #[cfg(not(target_arch = "wasm32"))]
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn hover(k: usize, t: &Take) -> String {
    let mut s = format!("{} · {}", k + 1, if t.label.is_empty() { "take" } else { &t.label });
    if let Some(u) = t.usd {
        s.push_str(&format!(" · {}", service::usd(u)));
    }
    let a = ago(t.when, now());
    if !a.is_empty() {
        s.push_str(&format!(" · {a}"));
    }
    if !t.commercial {
        s.push_str("\nmade by a model not free for commercial use");
    }
    s
}

/// The id a suite aims at: thumbnail `k` of `node`'s history.
pub fn thumb_id(node: u64, k: usize) -> egui::Id {
    egui::Id::new(("genai-take", node, k))
}

/// The id of one of the history's buttons: "settings", "load", "feed".
pub fn button_id(node: u64, what: &str) -> egui::Id {
    egui::Id::new(("genai-take-btn", node, what))
}

/// The history section of the node's panel.
pub(super) fn takes(ui: &mut Ui, node: u64, p: &mut GenAiParams, v: &View) -> Option<Action> {
    let mut act = None;
    if p.pipeline.takes.is_empty() {
        ui.weak(small("no takes yet: the node passes its input through"));
    } else {
        let dir = config::results_dir();
        let fixed = p.model.clone();
        let pl = &mut p.pipeline;
        let picked = pl.take.min(pl.takes.len() - 1);
        let mut forget = None;
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            ui.label(small(format!("takes {}/{}", picked + 1, pl.takes.len())));
            let key = pl.takes[picked].key.clone();
            let has_recipe = dir.join(&key).join(genai::history::RECIPE).exists();
            let r = ui
                .add_enabled(has_recipe, egui::Button::new(small("settings")).small())
                .on_hover_text("put back the prompt, seed and model that made this take (nothing runs)")
                .on_disabled_hover_text("this take was made before settings were kept with it");
            crate::widgets::note_rect(ui, button_id(node, "settings"), r.rect);
            if r.clicked() {
                if let Some(steps) = genai::history::load_recipe(&dir, &key) {
                    genai::history::restore(pl, &steps, fixed.as_deref());
                }
            }
            let on_disk = !v.take_missing;
            let r = ui
                .add_enabled(on_disk, egui::Button::new(small("load")).small())
                .on_hover_text("make a Load node of this take beside the node (a Splat node, for a 3D take)");
            crate::widgets::note_rect(ui, button_id(node, "load"), r.rect);
            if r.clicked() {
                act = Some(Action::TakeToLoad { take: picked, wire: false });
            }
            let r = ui
                .add_enabled(on_disk && !pl.takes[picked].is_splat(), egui::Button::new(small("feed")).small())
                .on_hover_text("use this take as the node's image input: the next run's reference");
            crate::widgets::note_rect(ui, button_id(node, "feed"), r.rect);
            if r.clicked() {
                act = Some(Action::TakeToLoad { take: picked, wire: true });
            }
            let sheet_id = sheet_id(node);
            let mut open: bool = ui.data(|d| d.get_temp(sheet_id)).unwrap_or(false);
            let r = ui
                .add(egui::Button::selectable(open, small("sheet")).small())
                .on_hover_text("the takes side by side, large: click one to show it");
            crate::widgets::note_rect(ui, button_id(node, "sheet"), r.rect);
            if r.clicked() {
                open = !open;
                ui.data_mut(|d| d.insert_temp(sheet_id, open));
            }
            if ui.add(egui::Button::new(small("×")).small()).on_hover_text("forget this take (its file stays on disk)").clicked() {
                forget = Some(picked);
            }
        });
        egui::ScrollArea::vertical().id_salt(("genai-takes", node)).max_height(THUMB_H * 2.0 + 6.0).show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(3.0, 3.0);
                // Newest first.
                for k in (0..pl.takes.len()).rev() {
                    let t = &pl.takes[k];
                    let tex = thumb(ui.ctx(), &t.path(&dir), false);
                    let w = match &tex {
                        Some(tex) => {
                            let s = tex.size_vec2();
                            (THUMB_H * s.x / s.y.max(1.0)).clamp(THUMB_H * 0.4, THUMB_H * 2.4)
                        }
                        None => THUMB_H * 16.0 / 9.0,
                    };
                    let (rect, r) = ui.allocate_exact_size(egui::vec2(w, THUMB_H), Sense::click());
                    crate::widgets::note_rect(ui, thumb_id(node, k), rect);
                    if ui.is_rect_visible(rect) {
                        let pt = ui.painter_at(rect);
                        match &tex {
                            Some(tex) => {
                                let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                                pt.image(tex.id(), rect, uv, Color32::WHITE);
                            }
                            None => {
                                pt.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
                                pt.text(
                                    rect.center(),
                                    egui::Align2::CENTER_CENTER,
                                    if t.is_splat() { "3D".to_string() } else { format!("{}", k + 1) },
                                    egui::FontId::proportional(10.0),
                                    ui.visuals().weak_text_color(),
                                );
                            }
                        }
                        if t.is_video() {
                            // A play triangle, drawn: the fonts have no glyph for one.
                            let c = rect.left_bottom() + egui::vec2(8.0, -8.0);
                            let tri = vec![c + egui::vec2(-3.0, -4.0), c + egui::vec2(4.0, 0.0), c + egui::vec2(-3.0, 4.0)];
                            pt.add(egui::Shape::convex_polygon(tri, Color32::WHITE, Stroke::new(1.0, Color32::BLACK)));
                        }
                        if !t.commercial {
                            pt.circle_filled(rect.right_top() + egui::vec2(-5.0, 5.0), 3.0, theme::cost_nc());
                        }
                        let stroke = if k == picked {
                            Stroke::new(2.0, ui.visuals().selection.stroke.color)
                        } else if r.hovered() {
                            Stroke::new(1.0, ui.visuals().widgets.hovered.fg_stroke.color)
                        } else {
                            Stroke::NONE
                        };
                        pt.rect_stroke(rect.shrink(1.0), 2.0, stroke, egui::StrokeKind::Inside);
                    }
                    if r.clicked() && k != pl.take {
                        pl.take = k;
                        act = Some(Action::Show);
                    }
                    r.on_hover_text(hover(k, t));
                }
            });
        });
        if let Some(k) = sheet(ui.ctx(), node, pl, &dir) {
            pl.take = k;
            act = Some(Action::Show);
        }
        if let Some(k) = forget {
            pl.takes.remove(k);
            if pl.take > k || pl.take >= pl.takes.len() {
                pl.take = pl.take.saturating_sub(1);
            }
            act = Some(Action::Show);
        }
        if v.take_missing {
            ui.colored_label(theme::warn(), small("this take's file is not on this machine; running again bills again"));
        }
        if pl.picked().is_some_and(|t| t.is_splat()) {
            ui.weak(small("a 3D take: the Splat node made from it shows it; this node passes its input through"));
        }
        if pl.picked().is_some_and(|t| t.is_video()) && !video_ok() {
            // Loud, never a silent black: the node passes its input through.
            ui.colored_label(theme::error(), small(NO_VIDEO));
        }
    }
    act
}

fn sheet_id(node: u64) -> egui::Id {
    egui::Id::new(("genai-sheet", node))
}

/// The contact sheet: the node's takes side by side, large, newest first,
/// each under what made it. A floating window, shut by default and by its
/// own close button (it would cover the viewer's gizmos if it lingered).
/// Returns the take clicked, if one was.
fn sheet(ctx: &egui::Context, node: u64, pl: &genai::Pipeline, dir: &Path) -> Option<usize> {
    let id = sheet_id(node);
    let mut open: bool = ctx.data(|d| d.get_temp(id)).unwrap_or(false);
    if !open {
        return None;
    }
    let was = open;
    let mut picked = None;
    egui::Window::new("GenAI takes")
        .id(id)
        .open(&mut open)
        .default_size([720.0, 520.0])
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);
                    for k in (0..pl.takes.len()).rev() {
                        let t = &pl.takes[k];
                        let tex = thumb(ui.ctx(), &t.path(dir), true);
                        let w = match &tex {
                            Some(tex) => {
                                let s = tex.size_vec2();
                                (SHEET_H * s.x / s.y.max(1.0)).clamp(SHEET_H * 0.4, SHEET_H * 2.4)
                            }
                            None => SHEET_H * 16.0 / 9.0,
                        };
                        ui.vertical(|ui| {
                            ui.set_width(w);
                            let (rect, r) = ui.allocate_exact_size(egui::vec2(w, SHEET_H), Sense::click());
                            crate::widgets::note_rect(ui, egui::Id::new(("genai-sheet-take", node, k)), rect);
                            let pt = ui.painter_at(rect);
                            match &tex {
                                Some(tex) => {
                                    let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
                                    pt.image(tex.id(), rect, uv, Color32::WHITE);
                                }
                                None => {
                                    pt.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
                                    let what = if t.is_splat() { "3D" } else { "…" };
                                    pt.text(
                                        rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        what,
                                        egui::FontId::proportional(14.0),
                                        ui.visuals().weak_text_color(),
                                    );
                                }
                            }
                            if k == pl.take {
                                let s = Stroke::new(2.0, ui.visuals().selection.stroke.color);
                                pt.rect_stroke(rect.shrink(1.0), 2.0, s, egui::StrokeKind::Inside);
                            }
                            if r.clicked() {
                                picked = Some(k);
                            }
                            let mut text = small(hover(k, t).lines().next().unwrap_or_default().to_string());
                            if !t.commercial {
                                text = text.color(theme::cost_nc());
                            }
                            ui.add(egui::Label::new(text).truncate());
                        });
                    }
                });
            });
        });
    if open != was {
        ctx.data_mut(|d| d.insert_temp(id, open));
    }
    picked.filter(|&k| k != pl.take)
}

impl crate::Gfx {
    /// Make a Load node of `node`'s take `k`, beside it; `wire` also feeds
    /// it into the node's image input (replacing whatever was there), so
    /// an earlier result becomes the next run's reference.
    pub(crate) fn genai_take_to_load(&mut self, node: u64, k: usize, wire: bool) {
        let Some((path, pos)) = self.egraph.node(node).and_then(|n| {
            let t = super::params(&n.kind)?.pipeline.takes.get(k)?;
            Some((t.path(&config::results_dir()), n.pos))
        }) else {
            return;
        };
        if !path.exists() {
            self.render_status = Some("that take's file is not on this machine".into());
            return;
        }
        if genai::pipeline::is_splat_file(&path.to_string_lossy()) {
            return self.genai_splat_node(node, &path);
        }
        let s = path.to_string_lossy().into_owned();
        let slot = self.next_read_slot;
        self.next_read_slot += 1;
        let at = if wire { pos + egui::vec2(-190.0, -140.0) } else { pos + egui::vec2(-190.0, 0.0) };
        let id = self.egraph.add(engine::graph::NodeKind::read(slot, &s), at);
        if let Some(n) = self.egraph.nodes.get_mut(&id) {
            n.tv = crate::editor::new_read_tv(&n.kind);
        }
        self.load_read(id, &s);
        if wire {
            self.egraph.connect(node, 0, id);
        }
        self.playback_dirty = true;
    }
}

impl crate::Gfx {
    /// A 3D take that just landed (or was picked again by a cached re-run)
    /// becomes a Splat node beside `node`. Once: a capture some Splat node
    /// already shows is left alone. Nothing happens for a picture.
    pub(crate) fn genai_take_to_splat(&mut self, node: u64, k: usize) {
        let Some(path) = self.egraph.node(node).and_then(|n| {
            let t = super::params(&n.kind)?.pipeline.takes.get(k)?;
            t.is_splat().then(|| t.path(&config::results_dir()))
        }) else {
            return;
        };
        let s = path.to_string_lossy();
        let shown = self.egraph.nodes.values().any(|n| {
            matches!(&n.kind, engine::graph::NodeKind::Splat(p) if p.path == s)
        });
        if !shown && path.exists() {
            self.genai_splat_node(node, &path);
        }
    }

    /// A Splat node of the capture at `path`, to the right of `node`.
    fn genai_splat_node(&mut self, node: u64, path: &Path) {
        let Some(pos) = self.egraph.node(node).map(|n| n.pos) else { return };
        let p = engine::graph::SplatParams { path: path.to_string_lossy().into_owned(), ..Default::default() };
        let id = self.egraph.add(engine::graph::NodeKind::Splat(p), pos + egui::vec2(190.0, 0.0));
        self.selected = vec![id];
        self.playback_dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ages_read_like_a_person_would_say_them() {
        assert_eq!(ago(0, 1000), "");
        assert_eq!(ago(1000, 1030), "just now");
        assert_eq!(ago(1000, 1000 + 600), "10 min ago");
        assert_eq!(ago(1000, 1000 + 3 * 3600), "3 h ago");
        assert_eq!(ago(1000, 1000 + 3 * 86400), "3 d ago");
        assert_eq!(ago(2000, 1000), "", "a clock that went backwards says nothing");
    }
}
