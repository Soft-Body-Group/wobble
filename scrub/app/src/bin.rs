//! The bin: the left panel, the list of clips this session has brought in,
//! grouped into bins (a post's own rushes, rushes by shoot day, clips
//! brought in by hand). Clips come in through the clip browser
//! (`clipbrowser`, seq_manip's, opened by "browse…" / cmd+I), a drop onto
//! the bin, or a post's rushes scan (`scan_rushes`).
//!
//! Keys while the panel has focus: up / down move the selection, enter
//! loads the item into the source viewer, backspace takes it out of the bin.
//!
//! Two views, kept in ui.json: a LIST, or TILES (a size slider beside the
//! switch). A tile shows the clip's poster; moving the pointer across it
//! scrubs the clip (Premiere / Resolve style) with a thin playhead line,
//! from tiny frames decoded off-thread on decoders of their own
//! (`binthumbs`), so it never touches the viewers' playback. Click, cmd
//! click and shift click select (several clips drag down onto the timeline
//! in the bin's order); double-click or Enter loads into the source.

use egui::{Align2, Color32, FontId, Key, Sense, Ui};
use imfram::gallery::files::{self, Kind};
use std::path::{Path, PathBuf};

const ROW_H: f32 = 20.0;

#[derive(Clone, Default)]
pub struct Item {
    pub path: PathBuf,
    pub name: String,
    pub detail: String,
    /// The bin it is in: "" for clips brought in by hand, "post: …" for a
    /// post's own rushes, "rushes/2026-11-11" for the rushes folders,
    /// `SELECTS` for subclips marked in selects mode.
    pub group: String,
    /// A SUBCLIP: the source frames in..out (both included, frame indices
    /// as the source marks count them). `None`: the whole file.
    pub range: Option<(i64, i64)>,
    /// Selects mode's verdict on a whole clip (subclips have none).
    pub rating: Option<Rating>,
    /// The in / out the clip last had in the source viewer, put back when
    /// it is loaded again (whole clips only).
    pub marks: model::Marks,
}

/// A clip's verdict in selects mode: S selects, R rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Rating {
    Select,
    Reject,
}

/// Which clips selects mode lists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Filter {
    #[default]
    All,
    Selects,
    Unmarked,
    Rejects,
}

impl Filter {
    pub const ALL: [Filter; 4] = [Filter::All, Filter::Selects, Filter::Unmarked, Filter::Rejects];
    pub fn name(self) -> &'static str {
        match self {
            Filter::All => "all",
            Filter::Selects => "selects",
            Filter::Unmarked => "unmarked",
            Filter::Rejects => "rejects",
        }
    }
    pub fn keeps(self, r: Option<Rating>) -> bool {
        match self {
            Filter::All => true,
            Filter::Selects => r == Some(Rating::Select),
            Filter::Unmarked => r.is_none(),
            Filter::Rejects => r == Some(Rating::Reject),
        }
    }
}

/// The bin subclips marked in selects mode go in (M there).
pub const SELECTS: &str = "selects";

/// What tells bin items apart: the file, and a subclip's range.
pub type ItemKey = (PathBuf, Option<(i64, i64)>);

impl Item {
    pub fn key(&self) -> ItemKey {
        (self.path.clone(), self.range)
    }
}

/// Past this many items the rushes bins open folded.
const FOLD_OVER: usize = 300;

/// The bin a post's own rushes go in.
pub const POST_PREFIX: &str = "post: ";

/// Rushes folders listed off the UI thread: (rushes folder, its label).
pub type RushesScan = std::sync::mpsc::Receiver<Vec<Item>>;

/// A folder's entries with only IMAGE sequences collapsed. imfram's scan
/// groups any numbered run of files, which would fold a day of numbered
/// movie rushes (`clip_0941.mp4`, `clip_0942.mp4`…) into one "sequence":
/// those are split back into their files here.
pub fn scan_dir(dir: &Path) -> Result<Vec<files::Entry>, String> {
    let mut out = Vec::new();
    for e in files::scan(dir, false)? {
        match &e.src.seq {
            Some(seq) if e.src.kind == Kind::Sequence && !media::sequence::is_image(&e.src.path) => {
                for &n in seq.frames.iter() {
                    let path = seq.path_of(n);
                    let md = std::fs::metadata(&path).ok();
                    let name = path.file_name().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
                    let movie = files::MOVIE_EXTS.contains(&path.extension().map(|x| x.to_string_lossy().to_lowercase()).unwrap_or_default().as_str());
                    out.push(files::Entry {
                        id: {
                            use std::hash::{Hash, Hasher};
                            let mut h = std::collections::hash_map::DefaultHasher::new();
                            path.hash(&mut h);
                            h.finish()
                        },
                        name,
                        size: md.as_ref().map_or(0, |m| m.len()),
                        modified: md.as_ref().and_then(|m| m.modified().ok()),
                        frames: 1,
                        hidden: false,
                        src: files::FileSrc { path, kind: if movie { Kind::Movie } else { Kind::File }, seq: None },
                    });
                }
            }
            _ => out.push(e),
        }
    }
    out.sort_by(|a, b| files::natural(&a.name, &b.name));
    Ok(out)
}

/// Media under `root` (folders `depth` deep), each in a bin named by its
/// folder relative to `label_base`, prefixed with `label`. Sequences
/// collapse to one item (imfram's folder scan); nothing is probed.
pub fn scan_tree(root: &Path, label_base: &Path, label: &str, depth: usize, out: &mut Vec<Item>) {
    let Ok(entries) = scan_dir(root) else { return };
    let rel = root.strip_prefix(label_base).map(|r| r.to_string_lossy().into_owned()).unwrap_or_default();
    let group = if label.is_empty() { rel } else if rel.is_empty() { label.to_string() } else { format!("{label} / {rel}") };
    for e in entries {
        match e.src.kind {
            Kind::Dir => {
                if depth > 0 {
                    scan_tree(&e.src.path, label_base, label, depth - 1, out);
                }
            }
            _ if is_media(&e) => {
                let detail = match e.src.kind {
                    Kind::Sequence => e.src.seq.as_ref().map_or(String::new(), |s| format!("{}-{}", s.first(), s.last())),
                    _ => files::size_text(e.size),
                };
                out.push(Item { path: e.src.frame_path(0), name: e.name, detail, group: group.clone(), ..Default::default() });
            }
            _ => {}
        }
    }
}

pub struct Bin {
    pub items: Vec<Item>,
    pub sel: usize,
    /// Where browsing for clips starts (the clip browser goes there when it
    /// changes: the first clip's folder, a post's rushes).
    pub dir: PathBuf,
    /// A rushes listing on its way.
    scan: Option<RushesScan>,
    /// Bins folded shut.
    folded: std::collections::HashSet<String>,
    /// Tiles instead of rows, and their width (kept by the editor in ui.json).
    pub tiles: bool,
    pub tile: f32,
    /// `tiles` / `tile` were read from ui.json.
    pub prefs_in: bool,
    /// Selected clips (click, cmd click, shift click); `sel` is the last.
    pub chosen: std::collections::HashSet<ItemKey>,
    /// Hover-scrub frames (made on first use).
    pub scrub: Option<crate::binthumbs::Scrub>,
    /// Last frame's tiles on screen (tests, drags).
    pub tile_rects: Vec<(PathBuf, egui::Rect)>,
}

/// What the panel asks the editor to do.
pub enum Action {
    None,
    /// Load this file (a movie, or a sequence's first frame) into the
    /// source viewer; a subclip with its range as the marks.
    Load(PathBuf, Option<(i64, i64)>),
    /// Open the clip browser (the "browse…" button).
    Browse,
}

fn is_media(e: &files::Entry) -> bool {
    match e.src.kind {
        Kind::Dir | Kind::Movie | Kind::Sequence => true,
        Kind::File => media::sequence::is_image(&e.src.path),
    }
}

impl Bin {
    pub fn new(dir: PathBuf) -> Self {
        Self { items: Vec::new(), sel: 0, dir, scan: None, folded: Default::default(), tiles: false, tile: 160.0, prefs_in: false, chosen: Default::default(), scrub: None, tile_rects: Vec::new() }
    }

    /// Where the next browse starts.
    pub fn go(&mut self, dir: PathBuf) {
        self.dir = dir;
    }

    /// Add to the bin (once per path) and select it.
    pub fn add(&mut self, path: &Path, detail: String) {
        self.add_to(path, detail, "");
    }

    /// The whole-clip item of `path` (not a subclip of it).
    pub fn clip(&self, path: &Path) -> Option<usize> {
        self.items.iter().position(|it| it.range.is_none() && it.path == path)
    }

    pub fn clip_mut(&mut self, path: &Path) -> Option<&mut Item> {
        self.items.iter_mut().find(|it| it.range.is_none() && it.path == path)
    }

    /// The display name of a file (a sequence as `name.#.exr`).
    pub fn name_of(path: &Path) -> String {
        match media::sequence::Pattern::of(path) {
            Some((p, _)) if media::sequence::is_image(path) => format!("{}#{}", p.prefix, p.suffix),
            _ => path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        }
    }

    /// A subclip of `path`, frames `inp..=out`, into the selects bin (as
    /// often as asked for the same clip; the very same range twice is
    /// refused). Its name numbers it among that clip's subclips. Returns
    /// the name, or `None` for a repeat.
    pub fn add_subclip(&mut self, path: &Path, inp: i64, out: i64, detail: String) -> Option<String> {
        let (inp, out) = (inp.min(out), inp.max(out));
        if self.items.iter().any(|it| it.path == path && it.range == Some((inp, out))) {
            return None;
        }
        let n = self.items.iter().filter(|it| it.path == path && it.range.is_some()).count() + 1;
        let name = format!("{} #{n}", Self::name_of(path));
        self.items.push(Item { path: path.to_path_buf(), name: name.clone(), detail, group: SELECTS.into(), range: Some((inp, out)), ..Default::default() });
        self.sort();
        self.sel = self.items.iter().position(|it| it.path == path && it.range == Some((inp, out))).unwrap_or(0);
        Some(name)
    }

    /// Whole clips (subclips left out) that `f` keeps, as item indices in
    /// the bin's order.
    pub fn listed(&self, f: Filter) -> Vec<usize> {
        (0..self.items.len()).filter(|&i| self.items[i].range.is_none() && f.keeps(self.items[i].rating)).collect()
    }

    /// The selects bin's items (subclips), as a range of item indices.
    pub fn selects_range(&self) -> std::ops::Range<usize> {
        let a = self.items.iter().position(|it| it.group == SELECTS).unwrap_or(self.items.len());
        let n = self.items[a..].iter().take_while(|it| it.group == SELECTS).count();
        a..a + n
    }

    /// Add to a bin (once per path: an item already in a bin stays there)
    /// and select it.
    pub fn add_to(&mut self, path: &Path, detail: String, group: &str) {
        if let Some(i) = self.clip(path) {
            // A post's own rush moves into the post's bin.
            if group.starts_with(POST_PREFIX) && self.items[i].group != group {
                self.items[i].group = group.to_string();
                self.sort();
            }
            self.sel = self.clip(path).unwrap_or(0);
            return;
        }
        let name = Self::name_of(path);
        self.items.push(Item { path: path.to_path_buf(), name, detail, group: group.to_string(), ..Default::default() });
        self.sort();
        self.sel = self.clip(path).unwrap_or(0);
    }

    /// Bins in order: the selects (subclips, in the order they were
    /// marked), the post's own, the rushes newest first, then clips
    /// brought in by hand; items by name within a bin, except the
    /// hand-brought ones, which keep the order they came in (an import of
    /// several clips lands in the tray's order).
    fn sort(&mut self) {
        fn rank(g: &str) -> u8 {
            if g == SELECTS {
                0
            } else if g.starts_with(POST_PREFIX) {
                1
            } else if g.is_empty() {
                3
            } else {
                2
            }
        }
        let arrival = |g: &str| g.is_empty() || g == SELECTS;
        self.items.sort_by(|a, b| {
            rank(&a.group)
                .cmp(&rank(&b.group))
                .then_with(|| if rank(&a.group) == 2 { b.group.cmp(&a.group) } else { a.group.cmp(&b.group) })
                .then_with(|| if arrival(&a.group) { std::cmp::Ordering::Equal } else { files::natural(&a.name, &b.name) })
        });
    }

    /// List rushes folders in the background; their bins replace the rushes
    /// bins there were. `dirs`: (rushes folder, label); a label is needed only
    /// when there are several marketing folders.
    pub fn scan_rushes(&mut self, dirs: Vec<(PathBuf, String)>) {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("rushes-scan".into())
            .spawn(move || {
                let mut out = Vec::new();
                for (rushes, label) in dirs {
                    let base = rushes.parent().map(Path::to_path_buf).unwrap_or_else(|| rushes.clone());
                    scan_tree(&rushes, &base, &label, 3, &mut out);
                }
                let _ = tx.send(out);
            })
            .ok();
        self.scan = Some(rx);
    }

    /// Take in a finished rushes listing: rushes bins replaced, the post's
    /// own and the hand-made ones kept (an item never in two bins).
    pub fn poll_scan(&mut self) -> bool {
        let Some(rx) = &self.scan else { return false };
        let Ok(found) = rx.try_recv() else { return false };
        self.scan = None;
        let keep = |g: &str| g.is_empty() || g == SELECTS || g.starts_with(POST_PREFIX);
        let cur = self.items.get(self.sel).map(|i| i.key());
        // A rush's verdict and remembered marks survive a re-listing.
        let mut was: std::collections::HashMap<PathBuf, (Option<Rating>, model::Marks)> = Default::default();
        for it in self.items.iter().filter(|i| i.range.is_none() && !keep(&i.group)) {
            was.insert(it.path.clone(), (it.rating, it.marks));
        }
        self.items.retain(|i| keep(&i.group));
        for mut it in found {
            if self.clip(&it.path).is_none() {
                if let Some((r, m)) = was.get(&it.path) {
                    (it.rating, it.marks) = (*r, *m);
                }
                self.items.push(it);
            }
        }
        self.sort();
        // A big library opens folded but for the newest day, so the panel
        // never lays out thousands of rows (a bin opens with a click).
        if self.items.len() > FOLD_OVER {
            let mut groups: Vec<String> = self.items.iter().map(|i| i.group.clone()).filter(|g| !g.is_empty() && g != SELECTS && !g.starts_with(POST_PREFIX)).collect();
            groups.dedup();
            for g in groups.into_iter().skip(1) {
                self.folded.insert(g);
            }
        }
        self.sel = cur.and_then(|c| self.items.iter().position(|i| i.key() == c)).unwrap_or(0);
        true
    }

    pub fn scanning(&self) -> bool {
        self.scan.is_some()
    }

    /// Keys, when the panel has focus: up / down, Enter loads, backspace
    /// takes the item out of the bin.
    pub fn keys(&mut self, ui: &mut Ui) -> Action {
        let none = egui::Modifiers::NONE;
        let take = |ui: &mut Ui, k: Key| crate::editor::take(ui, none, k, true);
        let n = self.items.len();
        if take(ui, Key::ArrowDown) && self.sel + 1 < n {
            self.sel += 1;
        }
        if take(ui, Key::ArrowUp) && self.sel > 0 {
            self.sel -= 1;
        }
        if take(ui, Key::Enter) {
            return self.items.get(self.sel).map_or(Action::None, |it| Action::Load(it.path.clone(), it.range));
        }
        if take(ui, Key::Backspace) && self.sel < self.items.len() {
            self.items.remove(self.sel);
            self.sel = self.sel.min(self.items.len().saturating_sub(1));
        }
        Action::None
    }

    pub fn ui(&mut self, ui: &mut Ui, focused: bool) -> Action {
        let mut act = Action::None;
        ui.horizontal(|ui| {
            imfram::dock::grip(ui);
            ui.strong(format!("bin ({})", self.items.len()));
            let b = ui.button("browse…  (cmd I)").on_hover_text("the clip browser (seq_manip's): Enter loads a clip into the source and the bin");
            if b.clicked() {
                act = Action::Browse;
            }
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tiles, false, "list");
            ui.selectable_value(&mut self.tiles, true, "tiles").on_hover_text("posters you scrub by moving the pointer across them");
            if self.tiles {
                ui.add(egui::Slider::new(&mut self.tile, 90.0..=320.0).show_value(false)).on_hover_text("tile size");
            }
        });
        ui.separator();
            self.poll_scan();
            if self.scanning() {
                ui.weak("listing rushes…");
            }
            if self.items.is_empty() && !self.scanning() {
                ui.weak("nothing yet: browse… (cmd I), then Enter or double-click a clip");
            }
            // Transitions: drag onto a cut on the timeline.
            let chip = ui
                .add(egui::Button::new(egui::RichText::new("× cross dissolve").small()).sense(Sense::click_and_drag()))
                .on_hover_text("drag onto a cut on the timeline (or cmd D / right-click a cut)");
            if chip.drag_started() {
                egui::DragAndDrop::set_payload(ui.ctx(), crate::timelineui::Payload::Dissolve);
            }
            let tchip = ui
                .add(egui::Button::new(egui::RichText::new("T text").small()).sense(Sense::click_and_drag()))
                .on_hover_text("drag onto a video track: a title there (or cmd shift T at the playhead)");
            if tchip.drag_started() {
                egui::DragAndDrop::set_payload(ui.ctx(), crate::timelineui::Payload::Text);
            }
            ui.separator();
            self.tile_rects.clear();
            if self.tiles {
                self.poll_scrub(&ui.ctx().clone());
            }
            egui::ScrollArea::vertical().id_salt("bin-items").auto_shrink([false, false]).show(ui, |ui| {
                let mut i = 0;
                while i < self.items.len() {
                    let g = self.items[i].group.clone();
                    let n = self.items[i..].iter().take_while(|x| x.group == g).count();
                    let shut = self.folded.contains(&g);
                    let title = if g.is_empty() { "clips".to_string() } else if g == SELECTS { "selects (subclips: M in selects mode)".to_string() } else { g.clone() };
                    let head = format!("{} {title}  ({n})", if shut { "+" } else { "-" });
                    let col = if g.starts_with(POST_PREFIX) { imfram::theme::highlight() } else { ui.visuals().strong_text_color() };
                    if ui.add(egui::Label::new(egui::RichText::new(head).color(col).small()).sense(Sense::click())).clicked() {
                        if shut {
                            self.folded.remove(&g);
                        } else {
                            self.folded.insert(g.clone());
                        }
                    }
                    if !shut {
                        if self.tiles {
                            self.tiles_ui(ui, i, n, focused, &mut act, false);
                        } else {
                            for k in i..i + n {
                                let (name, detail) = (self.items[k].name.clone(), self.items[k].detail.clone());
                                let on = k == self.sel || self.chosen.contains(&self.items[k].key());
                                let r = row(ui, &name, &detail, on, focused, false);
                                if let Some(rt) = self.items[k].rating {
                                    rating_badge(ui.painter(), egui::pos2(r.rect.right() - 12.0, r.rect.center().y), rt);
                                }
                                self.interact(ui, &r, k, &mut act, false);
                            }
                        }
                    }
                    i += n;
                }
            });
        act
    }
}

impl Bin {
    /// What a drag of the chosen items carries: plain files as before;
    /// with a subclip among them, every item with its range.
    pub fn payload(&self) -> Option<crate::timelineui::Payload> {
        use crate::timelineui::Payload;
        let many: Vec<ItemKey> = self.items.iter().filter(|it| self.chosen.contains(&it.key())).map(|it| it.key()).collect();
        Some(match many.as_slice() {
            [] => return None,
            [(p, None)] => Payload::File(p.clone()),
            _ if many.iter().all(|(_, r)| r.is_none()) => Payload::Files(many.into_iter().map(|(p, _)| p).collect()),
            _ => Payload::Clips(many),
        })
    }

    /// Click, cmd click (toggle), shift click (range), drag, double-click.
    /// `click_loads`: a single click loads it too (selects mode).
    fn interact(&mut self, ui: &Ui, r: &egui::Response, k: usize, act: &mut Action, click_loads: bool) {
        let key = self.items[k].key();
        if r.clicked() {
            // The click's own modifiers (the release event carries them).
            let m = ui.input(|i| i.events.iter().rev().find_map(|e| if let egui::Event::PointerButton { modifiers, .. } = e { Some(*modifiers) } else { None }).unwrap_or(i.modifiers));
            if m.command {
                if !self.chosen.remove(&key) {
                    self.chosen.insert(key.clone());
                }
            } else if m.shift {
                let (a, b) = (self.sel.min(k), self.sel.max(k));
                for it in &self.items[a..=b.min(self.items.len() - 1)] {
                    self.chosen.insert(it.key());
                }
            } else {
                self.chosen = [key.clone()].into();
                if click_loads {
                    *act = Action::Load(key.0.clone(), key.1);
                }
            }
            self.sel = k;
        }
        if r.drag_started() {
            if !self.chosen.contains(&key) {
                self.chosen = [key.clone()].into();
            }
            self.sel = k;
            if let Some(pl) = self.payload() {
                egui::DragAndDrop::set_payload(ui.ctx(), pl);
            }
        }
        if r.double_clicked() {
            *act = Action::Load(key.0, key.1);
        }
    }

    /// Hover-scrub frames: made on first use, polled once a frame.
    pub fn poll_scrub(&mut self, ctx: &egui::Context) {
        if self.scrub.get_or_insert_with(crate::binthumbs::Scrub::new).poll(ctx) {
            ctx.request_repaint();
        }
    }

    /// The selects bin alone, as tiles (selects mode's strip under the
    /// viewer): a single click loads a subclip with its marks.
    pub fn selects_ui(&mut self, ui: &mut Ui, focused: bool) -> Action {
        let mut act = Action::None;
        self.poll_scrub(&ui.ctx().clone());
        self.tile_rects.clear();
        let r = self.selects_range();
        if r.is_empty() {
            ui.weak("no subclips yet: mark in / out on a clip and press M (as often as you like per clip)");
            return act;
        }
        egui::ScrollArea::vertical().id_salt("selects-items").auto_shrink([false, false]).show(ui, |ui| {
            self.tiles_ui(ui, r.start, r.len(), focused, &mut act, true);
        });
        act
    }

    /// Items `i..i+n` as tiles, wrapped to the panel's width. A subclip's
    /// tile shows its in frame and scrubs only its in..out.
    fn tiles_ui(&mut self, ui: &mut Ui, i: usize, n: usize, focused: bool, act: &mut Action, click_loads: bool) {
        let w = self.tile.clamp(60.0, 400.0);
        let pic_h = (w * 9.0 / 16.0).round();
        let size = egui::vec2(w, pic_h + 18.0);
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 6.0);
            for k in i..i + n {
                let (rect, r) = ui.allocate_exact_size(size, Sense::click_and_drag());
                if !ui.is_rect_visible(rect) {
                    continue;
                }
                let path = self.items[k].path.clone();
                let range = self.items[k].range;
                let key = self.items[k].key();
                self.tile_rects.push((path.clone(), rect));
                let pic = egui::Rect::from_min_size(rect.min, egui::vec2(w, pic_h));
                let v = ui.visuals().clone();
                let p = ui.painter_at(rect);
                p.rect_filled(pic, 3.0, Color32::from_gray(12));
                let Some(scrub) = self.scrub.as_mut() else { break };
                // Hovered: scrub to the pointer; else the poster (a
                // subclip's: its in frame).
                let hover = r.hover_pos().filter(|q| pic.contains(*q) && !ui.input(|i| i.pointer.any_down()));
                let frac = hover.map(|q| (q.x - pic.left()) / pic.width().max(1.0));
                let (frame, tex, exact) = match (range, frac) {
                    (Some((a, b)), Some(x)) => scrub.at(&path, (a as f32 + x.clamp(0.0, 1.0) * (b - a) as f32).round().max(0.0) as u64),
                    (Some((a, _)), None) => {
                        let (_, t, _) = scrub.at(&path, a.max(0) as u64);
                        (None, t, true)
                    }
                    (None, Some(x)) => scrub.hover(&path, x),
                    (None, None) => (None, scrub.poster(&path), true),
                };
                if hover.is_some() && !exact {
                    ui.ctx().request_repaint_after(std::time::Duration::from_millis(8));
                }
                if let Some(t) = &tex {
                    let [tw, th] = t.size();
                    let s = (pic.width() / tw as f32).min(pic.height() / th as f32);
                    let r2 = egui::Rect::from_center_size(pic.center(), egui::vec2(tw as f32 * s, th as f32 * s));
                    p.image(t.id(), r2, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
                } else {
                    let msg = if scrub.broken(&path) { "?" } else { "…" };
                    p.text(pic.center(), Align2::CENTER_CENTER, msg, FontId::proportional(14.0), v.weak_text_color());
                }
                let info = scrub.info(&path);
                if let (Some(q), Some(f), Some(inf)) = (hover, frame, info) {
                    p.line_segment([egui::pos2(q.x, pic.top()), egui::pos2(q.x, pic.bottom())], egui::Stroke::new(1.5, imfram::theme::highlight()));
                    let tc = match range {
                        Some((a, b)) => format!("{}/{}", f as i64 - a + 1, b - a + 1),
                        None => format!("{f}/{}", inf.frames),
                    };
                    let at = egui::pos2(pic.right() - 4.0, pic.bottom() - 3.0);
                    let g = p.layout_no_wrap(tc, FontId::monospace(10.0), Color32::WHITE);
                    p.rect_filled(egui::Rect::from_min_size(at - g.size() - egui::vec2(3.0, 1.0), g.size() + egui::vec2(6.0, 2.0)), 2.0, Color32::from_black_alpha(160));
                    p.galley(at - g.size(), g, Color32::WHITE);
                }
                // A whole clip's verdict, in its corner.
                if let Some(rt) = self.items[k].rating {
                    rating_badge(&p, pic.left_top() + egui::vec2(9.0, 9.0), rt);
                }
                // Name and duration under the picture.
                let on = k == self.sel || self.chosen.contains(&key);
                let col = if on && focused { v.strong_text_color() } else { v.text_color() };
                let name_y = pic.bottom() + 9.0;
                let dur = info
                    .map(|i| match range {
                        Some((a, b)) => format!("{:.1}s", (b - a + 1) as f64 / i.fps.max(1.0)),
                        None if i.frames <= 1 => "still".to_string(),
                        None => format!("{:.1}s", i.frames as f64 / i.fps.max(1.0)),
                    })
                    .unwrap_or_default();
                let dg = p.layout_no_wrap(dur, FontId::proportional(10.0), v.weak_text_color());
                let dw = dg.size().x;
                p.galley(egui::pos2(rect.right() - dw - 2.0, name_y - dg.size().y / 2.0), dg, v.weak_text_color());
                let ng = p.layout(self.items[k].name.clone(), FontId::proportional(11.0), col, (w - dw - 8.0).max(10.0));
                let ng = if ng.rows.len() > 1 { p.layout_no_wrap(elide(&self.items[k].name, ((w - dw - 8.0) / 6.0) as usize), FontId::proportional(11.0), col) } else { ng };
                p.galley(egui::pos2(rect.left() + 2.0, name_y - 7.0), ng, col);
                if on {
                    let c = if focused { v.selection.bg_fill } else { v.widgets.inactive.fg_stroke.color };
                    p.rect_stroke(pic, 3.0, egui::Stroke::new(2.0, c), egui::StrokeKind::Inside);
                } else if r.hovered() {
                    p.rect_stroke(pic, 3.0, egui::Stroke::new(1.0, v.widgets.hovered.bg_stroke.color), egui::StrokeKind::Inside);
                }
                self.interact(ui, &r, k, act, click_loads);
            }
        });
    }
}

/// A select (green tick) or reject (red cross) mark, centred on `c`,
/// drawn as strokes (egui's fonts have no tick glyph).
pub fn rating_badge(p: &egui::Painter, c: egui::Pos2, r: Rating) {
    let w = egui::Stroke::new(1.6, Color32::WHITE);
    let v = |x: f32, y: f32| c + egui::vec2(x, y);
    match r {
        Rating::Select => {
            p.circle_filled(c, 7.0, Color32::from_rgb(46, 160, 67));
            p.line_segment([v(-3.5, 0.0), v(-1.0, 2.8)], w);
            p.line_segment([v(-1.0, 2.8), v(3.8, -2.8)], w);
        }
        Rating::Reject => {
            p.circle_filled(c, 7.0, Color32::from_rgb(200, 55, 55));
            p.line_segment([v(-3.0, -3.0), v(3.0, 3.0)], w);
            p.line_segment([v(-3.0, 3.0), v(3.0, -3.0)], w);
        }
    }
}

fn elide(s: &str, n: usize) -> String {
    let n = n.max(4);
    if s.chars().count() <= n {
        return s.to_string();
    }
    let keep: String = s.chars().take(n - 1).collect();
    format!("{keep}…")
}

fn row(ui: &mut Ui, name: &str, detail: &str, selected: bool, focused: bool, dir: bool) -> egui::Response {
    let (rect, r) = ui.allocate_exact_size(egui::vec2(ui.available_width(), ROW_H), Sense::click_and_drag());
    let v = ui.visuals();
    if selected {
        let fill = if focused { v.selection.bg_fill } else { v.widgets.inactive.bg_fill };
        ui.painter().rect_filled(rect, 2.0, fill);
    } else if r.hovered() {
        ui.painter().rect_filled(rect, 2.0, v.widgets.hovered.weak_bg_fill);
    }
    let p = ui.painter_at(rect);
    let col = if selected && focused { v.strong_text_color() } else { v.text_color() };
    let name = if dir { format!("{name}/") } else { name.to_string() };
    p.text(rect.left_center() + egui::vec2(6.0, 0.0), Align2::LEFT_CENTER, name, FontId::proportional(13.0), col);
    let dc: Color32 = v.weak_text_color();
    p.text(rect.right_center() - egui::vec2(6.0, 0.0), Align2::RIGHT_CENTER, detail, FontId::proportional(11.0), dc);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 5000 clips in 50 day folders plus sequences: the listing time
    /// (`cargo test --release -p app rushes_scan_speed -- --ignored --nocapture`).
    #[test]
    #[ignore]
    fn rushes_scan_speed() {
        let root = std::env::temp_dir().join(format!("scrub-scan-{}", std::process::id())).join("marketing");
        let rushes = root.join("rushes");
        for d in 0..50 {
            let day = rushes.join(format!("2026-10-{:02}", d % 28 + 1)).join(format!("cam{d}"));
            std::fs::create_dir_all(&day).unwrap();
            for c in 0..100 {
                std::fs::write(day.join(format!("clip{c:03}.mp4")), b"").unwrap();
            }
            for f in 1..=50 {
                std::fs::write(day.join(format!("seq.{f:04}.exr")), b"").unwrap();
            }
        }
        let t0 = std::time::Instant::now();
        let mut out = Vec::new();
        scan_tree(&rushes, &root, "", 3, &mut out);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!("listed {} items in {ms:.0} ms", out.len());
        assert_eq!(out.len(), 50 * 101, "every movie its own item, each image run one");
        let _ = std::fs::remove_dir_all(root.parent().unwrap());
    }

    #[test]
    fn the_posts_bin_comes_first_then_rushes_newest_first() {
        let mut b = Bin::new(PathBuf::from("/"));
        b.add_to(Path::new("/m/rushes/2026-11-10/a.mp4"), String::new(), "rushes/2026-11-10");
        b.add_to(Path::new("/m/rushes/2026-11-12/b.mp4"), String::new(), "rushes/2026-11-12");
        b.add_to(Path::new("/x/c.mov"), String::new(), "");
        b.add_to(Path::new("/m/rushes/2026-11-10/a.mp4"), String::new(), "post: Teaser");
        let groups: Vec<&str> = b.items.iter().map(|i| i.group.as_str()).collect();
        assert_eq!(groups, ["post: Teaser", "rushes/2026-11-12", ""]);
    }
}
