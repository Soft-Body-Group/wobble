//! Selects mode: going through the rushes before the edit.
//!
//! One viewer (the source viewer, big), the bin's clips listed down the
//! left with their verdicts, and the SELECTS bin of subclips along the
//! bottom. A single click on a clip views it. S selects it, R rejects it
//! (again: clears; U clears either), and 1 2 3 4 list all / selects /
//! unmarked / rejects. Up / down walk the list. Mark in / out and press M:
//! that range goes into the selects bin as a subclip, as often as you like
//! per clip (the clip's own in / out are remembered too, and come back
//! when it is viewed again).
//!
//! Back in edit mode (` switches) the subclips are ordinary bin items with
//! a range: double-click / Enter loads one into the source with its marks
//! (then , . or M), a drag lays it (or several, in order) onto the
//! timeline, and M with the bin focused appends the chosen items. Their
//! tiles scrub only their in..out.
//!
//! The verdicts, subclips and remembered marks are saved with the project
//! (format 5).

use crate::bin::{self, Filter, Rating};
use crate::editor::{take, Editor, Focus, Mode};
use egui::{pos2, vec2, Align2, FontId, Key, Modifiers, Rect, Sense, Stroke};
use imfram::hudkeys::{self, HudKey};
use model::Marks;
use std::path::{Path, PathBuf};

/// The key hints over the viewer in selects mode.
pub const KEYS: &[HudKey] = &[
    hudkeys::chord("click", "a clip: view it"),
    hudkeys::key("S", "select (again: clear)", Key::S),
    hudkeys::key("R", "reject (again: clear)", Key::R),
    hudkeys::key("U", "clear the verdict", Key::U),
    hudkeys::chord("1 2 3 4", "list all / selects / unmarked / rejects"),
    hudkeys::key("up down", "previous / next clip", Key::ArrowUp),
    hudkeys::key("I O", "mark in / out", Key::I),
    hudkeys::key("M", "in..out into the selects bin (again: another)", Key::M),
    hudkeys::chord("alt I O X", "clear in / out / both"),
    hudkeys::key("J K L", "shuttle (again: faster)", Key::K),
    hudkeys::key("space", "play / stop", Key::Space),
    hudkeys::key("left right", "frame step", Key::ArrowLeft),
    hudkeys::key("F", "fit the viewer", Key::F),
    hudkeys::chord("`", "back to edit"),
    hudkeys::chord("H  shift H", "key hints / readouts"),
];

const LIST_W: f32 = 300.0;
const ROW_H: f32 = 22.0;

impl Editor {
    /// ` : edit <-> selects.
    pub(crate) fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            Mode::Edit => Mode::Selects,
            Mode::Selects => Mode::Edit,
        };
        self.hold = None;
        if self.mode == Mode::Selects {
            // The keys drive the one viewer.
            self.focus = Focus::Source;
            self.monitor = Focus::Source;
            self.rec.transport.stop();
            self.say("selects mode: click a clip, S select, R reject, I O + M a subclip (` back to edit)");
        } else {
            self.say("edit mode: the selects bin's subclips go on the timeline by drag, Enter + , . or M (` selects)");
        }
    }

    /// The source clip's marks, kept on its bin item as its own (unless
    /// they are still what a drop or a subclip load put on).
    pub(crate) fn remember_source_marks(&mut self) {
        let Some(c) = &self.source else { return };
        if c.set_marks == Some(c.marks) {
            return;
        }
        let (p, m) = (c.original.clone(), c.marks);
        if let Some(it) = self.bin.clip_mut(&p) {
            it.marks = m;
        }
    }

    /// The marks to save for a bin item: the live ones of the clip in the
    /// source viewer, else the remembered ones.
    pub(crate) fn remembered_marks(&self, it: &bin::Item) -> Marks {
        if it.range.is_some() {
            return Marks::default();
        }
        match &self.source {
            Some(c) if c.original == it.path && c.set_marks != Some(c.marks) => c.marks,
            _ => it.marks,
        }
    }

    /// Load a bin item into the source viewer: a whole clip with its
    /// remembered marks, a subclip with its range as the marks and the
    /// playhead on its in.
    pub(crate) fn load_item(&mut self, p: &Path, range: Option<(i64, i64)>) {
        let keep = (self.bin.sel, self.bin.chosen.clone());
        self.open(p);
        if let (Some((a, b)), Some(c)) = (range, self.source.as_mut()) {
            if c.original == p {
                let m = Marks { inp: Some(a), out: Some(b) };
                c.marks = m;
                c.set_marks = Some(m);
                c.transport.stop();
                c.transport.seek(a.clamp(0, c.info.frames as i64 - 1) as u64);
            }
        }
        // Loading never moves the bin's own selection off what was clicked.
        (self.bin.sel, self.bin.chosen) = keep;
    }

    /// The source clip's verdict: `r`, or cleared if it already had it
    /// (`None` clears either).
    pub(crate) fn rate(&mut self, r: Option<Rating>) {
        let Some(p) = self.source.as_ref().map(|c| c.original.clone()) else {
            self.say("click a clip to view it first");
            return;
        };
        let Some(it) = self.bin.clip_mut(&p) else { return };
        it.rating = match r {
            Some(r) if it.rating == Some(r) => None,
            r => r,
        };
        let name = it.name.clone();
        let msg = match it.rating {
            Some(Rating::Select) => format!("selected {name}"),
            Some(Rating::Reject) => format!("rejected {name}"),
            None => format!("{name}: no verdict"),
        };
        self.say(msg);
    }

    /// M in selects mode: the source in..out (no in: from the first frame;
    /// no out: to the last) into the selects bin. An unrated clip becomes a
    /// select with it.
    pub(crate) fn add_subclip(&mut self) {
        let Some(c) = &self.source else {
            self.say("click a clip to view it first");
            return;
        };
        let last = c.info.frames as i64 - 1;
        let (a, b) = (c.marks.inp.unwrap_or(0).clamp(0, last), c.marks.out.unwrap_or(last).clamp(0, last));
        let (a, b) = (a.min(b), a.max(b));
        let fps = c.info.fps();
        let path = c.original.clone();
        let detail = format!("{} fr  {:.1}s", b - a + 1, (b - a + 1) as f64 / fps.max(1.0));
        let whole = c.marks.inp.is_none() && c.marks.out.is_none();
        match self.bin.add_subclip(&path, a, b, detail) {
            Some(name) => {
                if let Some(it) = self.bin.clip_mut(&path) {
                    if it.rating.is_none() {
                        it.rating = Some(Rating::Select);
                    }
                }
                self.say(format!("{name} into the selects bin ({} frames{})", b - a + 1, if whole { ", the whole clip: no marks" } else { "" }));
            }
            None => self.say("that in..out is in the selects bin already"),
        }
    }

    /// M in edit mode with the bin focused: the chosen bin items (the one
    /// under the cursor when none are), in bin order, appended to the end of
    /// the sequence on the targeted tracks; subclips by their range, clips
    /// whole. One undo step each.
    pub(crate) fn append_chosen(&mut self) {
        let mut picks: Vec<bin::ItemKey> = self.bin.items.iter().filter(|it| self.bin.chosen.contains(&it.key())).map(|it| it.key()).collect();
        if picks.is_empty() {
            picks.extend(self.bin.items.get(self.bin.sel).map(|it| it.key()));
        }
        if picks.is_empty() {
            self.say("nothing in the bin to append");
            return;
        }
        let keep = (self.bin.sel, self.bin.chosen.clone());
        let mut n = 0;
        for (p, r) in &picks {
            self.open(p);
            let Some(c) = self.source.as_mut().filter(|c| c.original == *p) else { break };
            let m = r.map_or(Marks::default(), |(a, b)| Marks { inp: Some(a), out: Some(b) });
            c.marks = m;
            c.set_marks = Some(m);
            c.transport.stop();
            c.transport.seek(r.map_or(0, |(a, _)| a.max(0) as u64));
            self.append();
            if self.error.is_some() {
                break;
            }
            n += 1;
        }
        (self.bin.sel, self.bin.chosen) = keep;
        self.focus = Focus::Bin;
        if n > 0 {
            self.say(format!("appended {n} clip(s) from the bin at the end (M)"));
        }
    }

    /// `SCRUB_SHOT_SELECTS`: selects mode with the source clip selected,
    /// two subclips of it marked, and in / out left on a third range.
    pub fn shot_selects(&mut self) {
        if self.mode != Mode::Selects {
            self.toggle_mode();
        }
        let Some(n) = self.source.as_ref().map(|c| c.info.frames as i64) else { return };
        for (a, b) in [(n / 8, n / 3), (n / 2, n * 3 / 4)] {
            if let Some(c) = &mut self.source {
                c.marks = Marks { inp: Some(a), out: Some(b) };
            }
            self.add_subclip();
        }
        if let Some(c) = &mut self.source {
            c.marks = Marks { inp: Some(n / 3), out: Some(n / 2) };
            c.transport.seek((n / 3).max(0) as u64);
        }
    }

    /// The clips selects mode lists now, in order.
    pub(crate) fn listed(&self) -> Vec<usize> {
        self.bin.listed(self.filter)
    }

    /// Up / down: the clip before / after the one in the viewer in the
    /// list (the first or last when it is not listed).
    fn walk(&mut self, d: i64) {
        let list = self.listed();
        if list.is_empty() {
            return;
        }
        let cur = self.source.as_ref().and_then(|c| self.bin.clip(&c.original)).and_then(|i| list.iter().position(|&x| x == i));
        let next = match cur {
            Some(k) => (k as i64 + d).clamp(0, list.len() as i64 - 1) as usize,
            None if d > 0 => 0,
            None => list.len() - 1,
        };
        let p = self.bin.items[list[next]].path.clone();
        self.load_item(&p, None);
        self.bin.sel = list[next];
    }

    /// The keys of selects mode (marks and the transport follow, as in edit).
    pub(crate) fn selects_keys(&mut self, ui: &mut egui::Ui) {
        let none = Modifiers::NONE;
        self.hold = None;
        self.focus = Focus::Source;
        self.monitor = Focus::Source;
        if take(ui, none, Key::S, false) {
            self.rate(Some(Rating::Select));
        }
        if take(ui, none, Key::R, false) {
            self.rate(Some(Rating::Reject));
        }
        if take(ui, none, Key::U, false) {
            self.rate(None);
        }
        if take(ui, none, Key::M, false) {
            self.add_subclip();
        }
        for (k, f) in [(Key::Num1, Filter::All), (Key::Num2, Filter::Selects), (Key::Num3, Filter::Unmarked), (Key::Num4, Filter::Rejects)] {
            if take(ui, none, k, false) {
                self.filter = f;
                self.say(format!("listing {}", f.name()));
            }
        }
        if take(ui, none, Key::ArrowUp, true) {
            self.walk(-1);
        }
        if take(ui, none, Key::ArrowDown, true) {
            self.walk(1);
        }
        if take(ui, none, Key::F, false) {
            self.views[0].fit();
        }
    }

    /// The main window in selects mode: the clip list down the left, the
    /// viewer, and the selects bin under it.
    pub(crate) fn selects_body(&mut self, ui: &mut egui::Ui, body: Rect) {
        let list_w = LIST_W.min(body.width() * 0.35);
        let strip_h = (body.height() * 0.3).clamp(140.0, 340.0);
        let list = Rect::from_min_max(body.min, pos2(body.min.x + list_w, body.max.y));
        let right = Rect::from_min_max(pos2(list.max.x + 4.0, body.min.y), body.max);
        let view = Rect::from_min_max(right.min, pos2(right.max.x, right.max.y - strip_h - 4.0));
        let strip = Rect::from_min_max(pos2(right.min.x, view.max.y + 4.0), right.max);
        #[cfg(feature = "clips")]
        self.bin_drop(ui, list);
        ui.scope_builder(egui::UiBuilder::new().max_rect(list.shrink(4.0)), |ui| self.clip_list(ui));
        ui.scope_builder(egui::UiBuilder::new().max_rect(view), |ui| self.tab_body(ui, crate::panes::Pane::Source));
        ui.scope_builder(egui::UiBuilder::new().max_rect(strip.shrink(4.0)), |ui| self.selects_strip(ui));
        let v = ui.visuals();
        for r in [list, strip] {
            ui.painter().rect_stroke(r, 2.0, Stroke::new(1.0, v.widgets.noninteractive.bg_stroke.color), egui::StrokeKind::Inside);
        }
    }

    /// The clips, with their verdicts, filtered.
    fn clip_list(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.strong("clips");
            if ui.button("browse…  (cmd I)").clicked() {
                self.import();
            }
        });
        let count = |f: Filter| self.bin.listed(f).len();
        let counts: Vec<(Filter, usize)> = Filter::ALL.iter().map(|&f| (f, count(f))).collect();
        ui.horizontal_wrapped(|ui| {
            for (k, (f, n)) in counts.iter().enumerate() {
                if ui.selectable_label(self.filter == *f, format!("{} {n}", f.name())).on_hover_text(format!("key {}", k + 1)).clicked() {
                    self.filter = *f;
                }
            }
        });
        ui.separator();
        self.bin.poll_scan();
        let list = self.listed();
        if list.is_empty() {
            let msg = if self.bin.items.iter().all(|i| i.range.is_some()) { "no clips yet: browse… (cmd I)" } else { "none listed here: 1 lists them all" };
            ui.weak(msg);
            return;
        }
        let cur: Option<PathBuf> = self.source.as_ref().map(|c| c.original.clone());
        let mut load = None;
        egui::ScrollArea::vertical().id_salt("selects-list").auto_shrink([false, false]).show(ui, |ui| {
            let mut group: Option<&str> = None;
            for &i in &list {
                let it = &self.bin.items[i];
                if group != Some(it.group.as_str()) {
                    group = Some(it.group.as_str());
                    let g = if it.group.is_empty() { "clips" } else { it.group.as_str() };
                    ui.label(egui::RichText::new(g).small().color(ui.visuals().weak_text_color()));
                }
                let (rect, r) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_H), Sense::click());
                if !ui.is_rect_visible(rect) {
                    continue;
                }
                let v = ui.visuals();
                let on = cur.as_deref() == Some(it.path.as_path());
                if on {
                    ui.painter().rect_filled(rect, 2.0, v.selection.bg_fill);
                } else if r.hovered() {
                    ui.painter().rect_filled(rect, 2.0, v.widgets.hovered.weak_bg_fill);
                }
                let p = ui.painter_at(rect);
                if let Some(rt) = it.rating {
                    bin::rating_badge(&p, pos2(rect.left() + 11.0, rect.center().y), rt);
                }
                let col = if on { v.strong_text_color() } else if it.rating == Some(Rating::Reject) { v.weak_text_color() } else { v.text_color() };
                p.text(pos2(rect.left() + 24.0, rect.center().y), Align2::LEFT_CENTER, &it.name, FontId::proportional(13.0), col);
                let subs = self.bin.items.iter().filter(|x| x.path == it.path && x.range.is_some()).count();
                let marked = it.marks != Marks::default();
                let tail = match (subs, marked) {
                    (0, false) => String::new(),
                    (0, true) => "in/out".into(),
                    (n, _) => format!("{n} sub"),
                };
                p.text(pos2(rect.right() - 6.0, rect.center().y), Align2::RIGHT_CENTER, tail, FontId::proportional(11.0), v.weak_text_color());
                if r.clicked() {
                    load = Some((it.path.clone(), i));
                }
            }
        });
        if let Some((p, i)) = load {
            self.load_item(&p, None);
            self.bin.sel = i;
        }
    }

    /// The selects bin along the bottom: scrubbable tiles of the subclips.
    fn selects_strip(&mut self, ui: &mut egui::Ui) {
        if !self.bin.prefs_in {
            (self.bin.tiles, self.bin.tile, self.bin.prefs_in) = (self.panes.bin_tiles, self.panes.bin_tile, true);
        }
        let n = self.bin.selects_range().len();
        ui.horizontal(|ui| {
            ui.strong(format!("selects bin ({n})"));
            ui.weak("M adds the marked range · double-click loads it · in edit mode drag onto the timeline");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add(egui::Slider::new(&mut self.bin.tile, 90.0..=320.0).show_value(false)).on_hover_text("tile size");
            });
        });
        if self.bin.tile != self.panes.bin_tile {
            self.panes.bin_tile = self.bin.tile;
            self.panes.changed();
        }
        ui.separator();
        match self.bin.selects_ui(ui, false) {
            bin::Action::Load(p, r) => self.load_item(&p, r),
            bin::Action::Browse => self.import(),
            bin::Action::None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny PNG sequence of `n` frames, `name.1001.png` on.
    fn sequence(dir: &Path, name: &str, n: usize) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        for i in 0..n {
            let f = std::fs::File::create(dir.join(format!("{name}.{:04}.png", 1001 + i))).unwrap();
            let mut e = png::Encoder::new(std::io::BufWriter::new(f), 16, 16);
            e.set_color(png::ColorType::Rgba);
            e.set_depth(png::BitDepth::Eight);
            e.write_header().unwrap().write_image_data(&vec![(i * 20) as u8; 16 * 16 * 4]).unwrap();
        }
        dir.join(format!("{name}.1001.png"))
    }

    #[test]
    fn subclips_ratings_and_remembered_marks() {
        let dir = std::env::temp_dir().join(format!("scrub-selects-{}", std::process::id()));
        let a = sequence(&dir.join("a"), "shotA", 8);
        let b = sequence(&dir.join("b"), "shotB", 6);
        let mut ed = Editor::new(true);
        ed.mode = Mode::Selects;
        ed.load_item(&a, None);
        assert!(ed.source.is_some(), "the png sequence loads");
        ed.rate(Some(Rating::Reject));
        ed.rate(Some(Rating::Select));
        assert_eq!(ed.bin.items[ed.bin.clip(&a).unwrap()].rating, Some(Rating::Select));
        ed.source.as_mut().unwrap().marks = Marks { inp: Some(2), out: Some(5) };
        ed.add_subclip();
        ed.add_subclip(); // the same range again: refused
        ed.source.as_mut().unwrap().marks = Marks { inp: Some(0), out: Some(3) };
        ed.add_subclip();
        let subs: Vec<_> = ed.bin.items.iter().filter(|i| i.range.is_some()).map(|i| (i.name.clone(), i.range)).collect();
        assert_eq!(subs.len(), 2, "{subs:?}");
        assert_eq!(subs[0].1, Some((2, 5)));
        assert_eq!(subs[1].1, Some((0, 3)));
        assert_eq!(ed.bin.selects_range(), 0..2, "the selects bin comes first");
        // Another clip, then back: the marks come back.
        ed.load_item(&b, None);
        assert_eq!(ed.source.as_ref().unwrap().marks, Marks::default());
        ed.load_item(&a, None);
        assert_eq!(ed.source.as_ref().unwrap().marks, Marks { inp: Some(0), out: Some(3) });
        // A subclip load sets its range, and leaving does not overwrite the
        // clip's own marks with it.
        ed.load_item(&a, Some((2, 5)));
        assert_eq!(ed.source.as_ref().unwrap().marks, Marks { inp: Some(2), out: Some(5) });
        ed.load_item(&b, None);
        assert_eq!(ed.bin.items[ed.bin.clip(&a).unwrap()].marks, Marks { inp: Some(0), out: Some(3) });
        // Filters.
        assert_eq!(ed.bin.listed(Filter::Selects).len(), 1);
        assert_eq!(ed.bin.listed(Filter::Unmarked).len(), 1);
        assert_eq!(ed.bin.listed(Filter::Rejects).len(), 0);
        // Edit mode: the bin's subclips append in order with their ranges.
        ed.mode = Mode::Edit;
        ed.bin.chosen = ed.bin.items[ed.bin.selects_range()].iter().map(|i| i.key()).collect();
        ed.append_chosen();
        let s = ed.sequence_summary();
        let v1: Vec<_> = s.clips.iter().filter(|c| c.0 == model::V1).collect();
        assert_eq!(v1.len(), 2, "{:?}", s.clips);
        // Saved and read back: the subclips, verdict and marks survive.
        let doc = ed.project_doc(None);
        let text = serde_json::to_string(&doc).unwrap();
        let back: crate::project::ProjectFile = serde_json::from_str(&text).unwrap();
        assert_eq!(back.bin.iter().filter(|e| e.range.is_some()).count(), 2);
        assert!(back.bin.iter().any(|e| e.rating == Some(Rating::Select) && e.marks == Marks { inp: Some(0), out: Some(3) }));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
