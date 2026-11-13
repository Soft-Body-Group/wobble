//! The setup browser: open, save and save as for `.wbl` setups, inside the
//! app rather than through the OS dialog.
//!
//! Three columns. **Places** on the left: home, the open comp's folder, the
//! bookmarks the sequence browser keeps, and the setups opened or saved
//! most recently (`Prefs::recent_setups`). The **list** in the middle:
//! folders, then setups, sortable by name / modified / size and narrowed by
//! the filter. A **preview** on the right of the picked setup: a miniature
//! of its node graph in the graph's own type colours, what it is made of,
//! its fps, range, format and the plates it reads — read straight from the
//! file, so nothing is loaded until you say so.
//!
//! Saving takes a name (`.wbl` is added), asks before replacing a file, and
//! can make a folder. Keys: ↑/↓ pick, Enter opens (a folder, or the setup),
//! Backspace goes up when no field is being typed into, Esc closes. While
//! it is open the app's hotkeys stand down.
//!
//! Native only: a page opens through its own picker and saves by download
//! (`webmain`), which is the right answer in a browser.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use egui::{pos2, vec2, Align, Color32, Id, Layout, Pos2, Rect, RichText, Sense, Stroke};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Open,
    Save,
}

#[derive(Debug, PartialEq)]
pub enum Action {
    None,
    Open(PathBuf),
    Save(PathBuf),
    Cancel,
}

/// What the app lends the side bar.
pub struct Places<'a> {
    pub bookmarks: &'a [PathBuf],
    pub recent: &'a [PathBuf],
    pub current: Option<&'a Path>,
}

#[derive(Clone, Debug)]
struct Entry {
    name: String,
    path: PathBuf,
    dir: bool,
    /// A `.json` setup from before the `.wbl` rename (open only).
    legacy: bool,
    modified: Option<SystemTime>,
    size: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sort {
    Name,
    Modified,
    Size,
}

/// What the preview column shows, read from the file without loading it.
#[derive(Default)]
struct Preview {
    /// Node centres (graph space) and their type colours.
    nodes: Vec<(Pos2, Color32)>,
    edges: Vec<(usize, usize)>,
    /// Node type -> how many, commonest first.
    kinds: Vec<(String, usize)>,
    fps: Option<f64>,
    range: Option<(i64, i64)>,
    canvas: Option<(u64, u64)>,
    plates: Vec<String>,
    err: Option<String>,
}

pub struct SetupBrowser {
    pub mode: Mode,
    dir: PathBuf,
    path_text: String,
    filter: String,
    /// Save mode: the file name being typed.
    name: String,
    entries: Vec<Entry>,
    /// By path, so a re-list keeps the pick.
    sel: Option<PathBuf>,
    sort: Sort,
    desc: bool,
    error: Option<String>,
    preview: Option<(PathBuf, Option<SystemTime>, Preview)>,
    /// A save that would replace this file, waiting for a yes.
    confirm: Option<PathBuf>,
    /// The new-folder field, while open.
    new_folder: Option<String>,
    /// Focus the new-folder field on its first pass (once: a request every
    /// pass would take back the focus Enter gives up, so it never committed).
    focus_folder: bool,
    /// Focus the name field (save) on the first pass.
    first: bool,
    /// Scroll the pick into view on the next pass (a key moved it).
    reveal: bool,
}

const ROW_H: f32 = 22.0;
const PLACES_W: f32 = 160.0;
const PREVIEW_W: f32 = 260.0;
const DATE_W: f32 = 92.0;
const SIZE_W: f32 = 64.0;

fn home() -> PathBuf {
    engine::paths::expand_home("~")
}

/// Unix seconds, negative before the epoch.
fn unix_secs(t: SystemTime) -> i64 {
    match t.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(e) => -(e.duration().as_secs() as i64),
    }
}

/// "just now", "12m ago", "5h ago", "3d ago", else the date (UTC).
fn fmt_age(t: Option<SystemTime>, now: SystemTime) -> String {
    let Some(t) = t else { return "—".into() };
    let s = unix_secs(now) - unix_secs(t);
    match s {
        i64::MIN..=59 => "just now".into(),
        60..=3599 => format!("{}m ago", s / 60),
        3600..=86_399 => format!("{}h ago", s / 3600),
        86_400..=604_799 => format!("{}d ago", s / 86_400),
        _ => {
            // Hinnant's civil-from-days, as the sequence browser does it.
            let days = unix_secs(t).div_euclid(86_400);
            let z = days + 719_468;
            let era = z.div_euclid(146_097);
            let doe = z.rem_euclid(146_097);
            let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let d = doy - (153 * mp + 2) / 5 + 1;
            let m = if mp < 10 { mp + 3 } else { mp - 9 };
            let y = yoe + era * 400 + i64::from(m <= 2);
            format!("{y:04}-{m:02}-{d:02}")
        }
    }
}

fn fmt_size(b: u64) -> String {
    match b {
        0..=1023 => format!("{b} B"),
        1024..=1_048_575 => format!("{:.0} KB", b as f64 / 1024.0),
        _ => format!("{:.1} MB", b as f64 / 1_048_576.0),
    }
}

/// The setups and folders in `dir`. Dotfiles are left out; `.json` is
/// listed only when opening (setups from before the `.wbl` rename).
fn list(dir: &Path, mode: Mode) -> std::io::Result<Vec<Entry>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let path = e.path();
        let Ok(meta) = std::fs::metadata(&path) else { continue };
        let dir = meta.is_dir();
        let ext = path.extension().and_then(|x| x.to_str()).map(str::to_ascii_lowercase);
        let legacy = ext.as_deref() == Some("json");
        if !dir && ext.as_deref() != Some("wbl") && !(legacy && mode == Mode::Open) {
            continue;
        }
        out.push(Entry { name, path, dir, legacy, modified: meta.modified().ok(), size: meta.len() });
    }
    Ok(out)
}

/// Folders first by name, then setups by the chosen key.
fn sort_entries(v: &mut [Entry], sort: Sort, desc: bool) {
    v.sort_by(|a, b| {
        b.dir.cmp(&a.dir).then_with(|| {
            if a.dir {
                return a.name.to_lowercase().cmp(&b.name.to_lowercase());
            }
            let o = match sort {
                Sort::Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
                Sort::Modified => a.modified.cmp(&b.modified),
                Sort::Size => a.size.cmp(&b.size),
            };
            if desc {
                o.reverse()
            } else {
                o
            }
        })
    });
}

/// The file a typed save name means: relative to `dir` unless it is a path
/// of its own (`~` expanded), always ending in `.wbl`.
fn resolve_name(dir: &Path, name: &str) -> Option<PathBuf> {
    let name = name.trim();
    if name.is_empty() || name == ".wbl" {
        return None;
    }
    let p = if name.starts_with('~') || Path::new(name).is_absolute() {
        engine::paths::expand_home(name)
    } else {
        dir.join(name)
    };
    let wbl = p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wbl"));
    Some(if wbl { p } else { PathBuf::from(format!("{}.wbl", p.display())) })
}

/// A typed folder name, checked against the strictest filesystem we run on
/// (Windows): one plain component, none of `\ / : * ? " < > |`, no trailing
/// dot or space (Windows silently drops them), not a reserved device name.
fn folder_name(text: &str) -> Result<&str, String> {
    let n = text.trim();
    if n.is_empty() || n == "." || n == ".." {
        return Err("type a folder name".into());
    }
    if let Some(c) = n.chars().find(|c| matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|') || c.is_control()) {
        return Err(format!("a folder name cannot contain {c:?}"));
    }
    if n.ends_with('.') {
        return Err("a folder name cannot end in a dot".into());
    }
    let stem = n.split('.').next().unwrap_or(n).trim_end().to_ascii_uppercase();
    let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.as_bytes()[3].is_ascii_digit());
    if reserved {
        return Err(format!("{n} is a reserved name on Windows"));
    }
    Ok(n)
}

/// A node's type name out of its saved `kind` (a bare string for a
/// field-less kind, else a one-key object).
fn kind_key(kind: &serde_json::Value) -> Option<String> {
    match kind {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(m) => m.keys().next().cloned(),
        _ => None,
    }
}

fn preview_of(path: &Path) -> Preview {
    let mut p = Preview::default();
    let v: serde_json::Value = match std::fs::read(path)
        .map_err(|e| e.to_string())
        .and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string()))
    {
        Ok(v) => v,
        Err(e) => {
            p.err = Some(e);
            return p;
        }
    };
    p.fps = v.get("fps").and_then(|f| f.as_f64());
    p.range = v.get("range").and_then(|r| Some((r.get(0)?.as_i64()?, r.get(1)?.as_i64()?)));
    p.canvas = v.get("canvas").and_then(|r| Some((r.get(0)?.as_u64()?, r.get(1)?.as_u64()?)));
    let Some(nodes) = v.pointer("/graph/nodes").and_then(|n| n.as_object()) else {
        p.err = Some("not a wobble setup".into());
        return p;
    };
    let mut index = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (id, n) in nodes {
        let kind = n.get("kind").cloned().unwrap_or_default();
        let parsed: Option<engine::graph::NodeKind> = serde_json::from_value(kind.clone()).ok();
        let name = parsed
            .as_ref()
            .map(|k| k.name().to_string())
            .or_else(|| kind_key(&kind))
            .unwrap_or_else(|| "?".into());
        // Its own colour where this build knows the kind; a node it does
        // not (a newer build's, a missing plugin) stays neutral.
        let colour = parsed.as_ref().map_or(Color32::GRAY, crate::editor::type_color);
        let pos = n
            .get("pos")
            .and_then(|q| Some(pos2(q.get("x")?.as_f64()? as f32, q.get("y")?.as_f64()? as f32)))
            .unwrap_or(Pos2::ZERO);
        if let Some(path) = kind.pointer("/Read/path").and_then(|s| s.as_str()) {
            let leaf = Path::new(path)
                .file_name()
                .map_or(path.to_string(), |f| f.to_string_lossy().into_owned());
            if !leaf.is_empty() && !p.plates.contains(&leaf) {
                p.plates.push(leaf);
            }
        }
        if let Ok(id) = id.parse::<u64>() {
            index.insert(id, p.nodes.len());
        }
        p.nodes.push((pos, colour));
        *counts.entry(name).or_default() += 1;
    }
    // Edges are [dst, input, src].
    for e in v.pointer("/graph/edges").and_then(|e| e.as_array()).into_iter().flatten() {
        let (Some(a), Some(b)) = (e.get(0).and_then(|x| x.as_u64()), e.get(2).and_then(|x| x.as_u64())) else {
            continue;
        };
        if let (Some(&a), Some(&b)) = (index.get(&a), index.get(&b)) {
            p.edges.push((a, b));
        }
    }
    let mut kinds: Vec<_> = counts.into_iter().collect();
    kinds.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    p.kinds = kinds;
    p
}

impl SetupBrowser {
    /// Open on `dir` (the comp's folder, usually), with `name` in the save
    /// field.
    pub fn new(mode: Mode, dir: Option<&Path>, name: &str) -> Self {
        let dir = dir.filter(|d| d.is_dir()).map(Path::to_path_buf).unwrap_or_else(home);
        let mut b = Self {
            mode,
            path_text: dir.display().to_string(),
            dir,
            filter: String::new(),
            name: name.to_string(),
            entries: Vec::new(),
            sel: None,
            sort: Sort::Modified,
            desc: true,
            error: None,
            preview: None,
            confirm: None,
            new_folder: None,
            focus_folder: false,
            first: true,
            reveal: false,
        };
        b.relist();
        b
    }

    /// Say why the last open / save failed, in the browser.
    pub fn fail(&mut self, e: String) {
        self.confirm = None;
        self.error = Some(e);
    }

    fn relist(&mut self) {
        match list(&self.dir, self.mode) {
            Ok(mut v) => {
                sort_entries(&mut v, self.sort, self.desc);
                self.entries = v;
                self.error = None;
            }
            Err(e) => {
                self.entries.clear();
                self.error = Some(format!("{}: {e}", self.dir.display()));
            }
        }
        if self.sel.as_ref().is_some_and(|s| !self.entries.iter().any(|e| &e.path == s)) {
            self.sel = None;
        }
    }

    fn navigate(&mut self, p: PathBuf) {
        if p.is_dir() {
            self.dir = p;
            self.path_text = self.dir.display().to_string();
            self.filter.clear();
            self.sel = None;
            self.confirm = None;
            self.relist();
        } else if p.is_file() {
            if let Some(parent) = p.parent() {
                self.navigate(parent.to_path_buf());
            }
            self.pick(Some(p));
        } else {
            self.error = Some(format!("no such folder: {}", p.display()));
            self.path_text = self.dir.display().to_string();
        }
    }

    fn up(&mut self) {
        if let Some(parent) = self.dir.parent() {
            let was = self.dir.clone();
            self.navigate(parent.to_path_buf());
            self.sel = Some(was);
            self.reveal = true;
        }
    }

    /// Pick a row. In save mode a picked setup's name lands in the field.
    fn pick(&mut self, p: Option<PathBuf>) {
        if self.mode == Mode::Save {
            if let Some(f) = p.as_ref().filter(|p| p.is_file()).and_then(|p| p.file_name()) {
                self.name = f.to_string_lossy().into_owned();
            }
        }
        self.confirm = None;
        self.sel = p;
    }

    fn visible(&self) -> Vec<&Entry> {
        let f = self.filter.trim().to_lowercase();
        self.entries.iter().filter(|e| f.is_empty() || e.name.to_lowercase().contains(&f)).collect()
    }

    /// Enter / double-click / the button: into a folder, or finish.
    fn activate(&mut self) -> Action {
        if let Some(sel) = self.sel.clone() {
            if sel.is_dir() {
                self.navigate(sel);
                return Action::None;
            }
            if self.mode == Mode::Open {
                return Action::Open(sel);
            }
        }
        if self.mode == Mode::Open {
            return Action::None;
        }
        let Some(path) = resolve_name(&self.dir, &self.name) else {
            self.error = Some("type a name to save as".into());
            return Action::None;
        };
        if path.exists() && self.confirm.as_ref() != Some(&path) {
            self.confirm = Some(path);
            return Action::None;
        }
        Action::Save(path)
    }

    pub fn show(&mut self, ctx: &egui::Context, places: &Places) -> Action {
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape)) {
            return Action::Cancel;
        }
        let mut act = Action::None;
        let mut open = true;
        let title = match self.mode {
            Mode::Open => "open setup",
            Mode::Save => "save setup",
        };
        let screen = ctx.content_rect();
        let size = vec2(900.0_f32.min(screen.width() - 40.0), 520.0_f32.min(screen.height() - 60.0));
        egui::Window::new(title)
            .id(Id::new("setup-browser"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .fixed_size(size)
            .default_pos(screen.center() - size * 0.5)
            .show(ctx, |ui| act = self.body(ui, places));
        if !open {
            return Action::Cancel;
        }
        act
    }

    fn body(&mut self, ui: &mut egui::Ui, places: &Places) -> Action {
        let mut act = Action::None;
        let typing = ui.ctx().egui_wants_keyboard_input();

        // Keys first, so the text fields never see ↑/↓.
        let (up_k, down_k, enter_k, back_k) = ui.ctx().input_mut(|i| {
            use egui::{Key, Modifiers};
            (
                i.consume_key(Modifiers::NONE, Key::ArrowUp),
                i.consume_key(Modifiers::NONE, Key::ArrowDown),
                !typing && i.consume_key(Modifiers::NONE, Key::Enter),
                !typing && i.consume_key(Modifiers::NONE, Key::Backspace),
            )
        });
        if up_k || down_k {
            let vis: Vec<PathBuf> = self.visible().iter().map(|e| e.path.clone()).collect();
            if !vis.is_empty() {
                let at = self.sel.as_ref().and_then(|s| vis.iter().position(|p| p == s));
                let next = match (at, down_k) {
                    (None, true) => 0,
                    (None, false) => vis.len() - 1,
                    (Some(i), true) => (i + 1).min(vis.len() - 1),
                    (Some(i), false) => i.saturating_sub(1),
                };
                self.pick(Some(vis[next].clone()));
                self.reveal = true;
            }
        }
        if back_k {
            self.up();
        }
        if enter_k {
            act = self.activate();
        }

        // The path bar.
        ui.horizontal(|ui| {
            if ui
                .add_enabled(self.dir.parent().is_some(), egui::Button::new("⏶"))
                .on_hover_text("up a folder (Backspace)")
                .clicked()
            {
                self.up();
            }
            if ui.button("⟲").on_hover_text("look again").clicked() {
                self.relist();
            }
            let filter_w = 170.0;
            let w = (ui.available_width() - filter_w - 12.0).max(120.0);
            let r = ui.add(egui::TextEdit::singleline(&mut self.path_text).desired_width(w));
            if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                let p = engine::paths::expand_home(self.path_text.trim());
                self.navigate(p);
            }
            let f = ui.add(
                egui::TextEdit::singleline(&mut self.filter).hint_text("filter").desired_width(filter_w),
            );
            if f.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                act = self.activate();
            }
        });
        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }
        ui.separator();

        // The columns take what the fixed window leaves above the bottom bar.
        let bottom_h = 40.0;
        let h = (ui.available_height() - bottom_h).max(120.0);
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(vec2(PLACES_W, h), Layout::top_down(Align::Min), |ui| {
                ui.set_min_size(vec2(PLACES_W, h));
                egui::ScrollArea::vertical().id_salt("setup-places").show(ui, |ui| {
                    if let Some(a) = self.places_ui(ui, places) {
                        act = a;
                    }
                });
            });
            ui.separator();
            let mid_w = (ui.available_width() - PREVIEW_W - 16.0).max(200.0);
            ui.allocate_ui_with_layout(vec2(mid_w, h), Layout::top_down(Align::Min), |ui| {
                ui.set_min_size(vec2(mid_w, h));
                if let Some(a) = self.list_ui(ui, mid_w) {
                    act = a;
                }
            });
            ui.separator();
            ui.allocate_ui_with_layout(vec2(PREVIEW_W, h), Layout::top_down(Align::Min), |ui| {
                ui.set_min_size(vec2(PREVIEW_W, h));
                ui.set_max_width(PREVIEW_W);
                self.preview_ui(ui);
            });
        });

        ui.separator();
        ui.horizontal(|ui| {
            if let Some(a) = self.bottom_ui(ui) {
                act = a;
            }
        });
        self.first = false;
        act
    }

    fn places_ui(&mut self, ui: &mut egui::Ui, places: &Places) -> Option<Action> {
        let mut act = None;
        let heading = |ui: &mut egui::Ui, s: &str| {
            ui.add_space(2.0);
            ui.label(RichText::new(s).weak().small());
        };
        let mut go = None;
        let place = |ui: &mut egui::Ui, label: String, p: &Path, go: &mut Option<PathBuf>| {
            let on = self.dir == p;
            if ui
                .add(egui::Button::selectable(on, label).truncate())
                .on_hover_text(p.display().to_string())
                .clicked()
            {
                *go = Some(p.to_path_buf());
            }
        };
        heading(ui, "places");
        place(ui, "home".into(), &home(), &mut go);
        if let Some(cur) = places.current.and_then(Path::parent) {
            place(ui, "this comp's folder".into(), cur, &mut go);
        }
        for b in places.bookmarks {
            let name = b.file_name().map_or(b.display().to_string(), |n| n.to_string_lossy().into_owned());
            place(ui, name, b, &mut go);
        }
        if !places.recent.is_empty() {
            ui.add_space(6.0);
            heading(ui, "recent setups");
            for r in places.recent {
                let name = r.file_name().map_or(r.display().to_string(), |n| n.to_string_lossy().into_owned());
                let gone = !r.exists();
                let text = if gone { RichText::new(name).weak().strikethrough() } else { RichText::new(name) };
                let resp = ui
                    .add_enabled(!gone, egui::Button::selectable(self.sel.as_ref() == Some(r), text).truncate())
                    .on_hover_text(r.display().to_string());
                if resp.clicked() {
                    go = Some(r.clone());
                }
                if resp.double_clicked() && self.mode == Mode::Open {
                    act = Some(Action::Open(r.clone()));
                }
            }
        }
        if let Some(p) = go {
            self.navigate(p);
            self.reveal = true;
        }
        act
    }

    fn list_ui(&mut self, ui: &mut egui::Ui, w: f32) -> Option<Action> {
        let mut act = None;
        // The header sorts.
        ui.horizontal(|ui| {
            let mut head = |ui: &mut egui::Ui, s: Sort, label: &str, width: f32| {
                let on = self.sort == s;
                let arrow = if on { if self.desc { " ⏷" } else { " ⏶" } } else { "" };
                let r = ui.add_sized(
                    [width, 18.0],
                    egui::Button::new(RichText::new(format!("{label}{arrow}")).small()).frame(false),
                );
                if r.clicked() {
                    if on {
                        self.desc = !self.desc;
                    } else {
                        self.sort = s;
                        self.desc = s != Sort::Name;
                    }
                    sort_entries(&mut self.entries, self.sort, self.desc);
                }
            };
            head(ui, Sort::Name, "name", (w - DATE_W - SIZE_W - 24.0).max(60.0));
            head(ui, Sort::Modified, "modified", DATE_W);
            head(ui, Sort::Size, "size", SIZE_W);
        });
        let now = SystemTime::now();
        let rows: Vec<Entry> = self.visible().into_iter().cloned().collect();
        let reveal = std::mem::take(&mut self.reveal);
        egui::ScrollArea::vertical().id_salt("setup-list").auto_shrink([false, false]).show(ui, |ui| {
            if rows.is_empty() {
                ui.add_space(8.0);
                ui.weak(if self.filter.trim().is_empty() {
                    "no setups or folders here"
                } else {
                    "nothing matches the filter"
                });
            }
            for e in &rows {
                let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), ROW_H), Sense::click());
                let on = self.sel.as_ref() == Some(&e.path);
                let vis = ui.visuals();
                let p = ui.painter();
                if on {
                    p.rect_filled(rect, 0.0, vis.selection.bg_fill);
                } else if resp.hovered() {
                    p.rect_filled(rect, 0.0, vis.widgets.hovered.weak_bg_fill);
                }
                let ink = if on { vis.selection.stroke.color } else { vis.text_color() };
                let weak = if on { ink } else { vis.weak_text_color() };
                // A folder or a little graph, drawn: the fonts have neither.
                let ic = pos2(rect.left() + 12.0, rect.center().y);
                if e.dir {
                    let b = Rect::from_center_size(ic + vec2(0.0, 1.0), vec2(14.0, 9.0));
                    p.rect_filled(b, 1.0, weak.gamma_multiply(0.7));
                    p.rect_filled(Rect::from_min_size(b.left_top() - vec2(0.0, 2.5), vec2(6.0, 3.0)), 1.0, weak.gamma_multiply(0.7));
                } else {
                    let a = ic + vec2(-4.5, -3.0);
                    let b = ic + vec2(4.5, 0.0);
                    let c = ic + vec2(-4.5, 3.5);
                    p.line_segment([a, b], Stroke::new(1.0, weak));
                    p.line_segment([c, b], Stroke::new(1.0, weak));
                    for q in [a, b, c] {
                        p.circle_filled(q, 2.2, ink);
                    }
                }
                let font = egui::FontId::proportional(13.0);
                let name_r = Rect::from_min_max(
                    pos2(rect.left() + 26.0, rect.top()),
                    pos2(rect.right() - DATE_W - SIZE_W - 8.0, rect.bottom()),
                );
                let mut name = e.name.clone();
                if e.legacy {
                    name.push_str("  · old .json");
                }
                p.with_clip_rect(name_r).text(
                    pos2(name_r.left(), rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    name,
                    font.clone(),
                    ink,
                );
                if !e.dir {
                    let mono = egui::FontId::monospace(11.0);
                    p.text(
                        pos2(rect.right() - SIZE_W - 8.0, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        fmt_age(e.modified, now),
                        mono.clone(),
                        weak,
                    );
                    p.text(pos2(rect.right() - 4.0, rect.center().y), egui::Align2::RIGHT_CENTER, fmt_size(e.size), mono, weak);
                }
                if reveal && on {
                    resp.scroll_to_me(Some(Align::Center));
                }
                if resp.clicked() {
                    self.pick(Some(e.path.clone()));
                }
                if resp.double_clicked() {
                    self.pick(Some(e.path.clone()));
                    act = Some(self.activate());
                }
            }
        });
        act
    }

    fn preview_ui(&mut self, ui: &mut egui::Ui) {
        let Some(sel) = self.sel.clone().filter(|p| p.is_file()) else {
            ui.add_space(8.0);
            ui.weak(match self.mode {
                Mode::Open => "pick a setup to see what is in it",
                Mode::Save => "saves the comp as it is now",
            });
            return;
        };
        // Read once per file + modification time.
        let mtime = std::fs::metadata(&sel).and_then(|m| m.modified()).ok();
        if !self.preview.as_ref().is_some_and(|(p, t, _)| *p == sel && *t == mtime) {
            self.preview = Some((sel.clone(), mtime, preview_of(&sel)));
        }
        let Some((_, _, pv)) = &self.preview else { return };
        let name = sel.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        ui.label(RichText::new(name).strong());
        if let Some(e) = &pv.err {
            ui.colored_label(ui.visuals().error_fg_color, e);
            return;
        }

        // The graph in miniature, fitted to the box.
        let (rect, _) = ui.allocate_exact_size(vec2(PREVIEW_W - 8.0, 150.0), Sense::hover());
        let p = ui.painter().with_clip_rect(rect);
        p.rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
        if !pv.nodes.is_empty() {
            let b = Rect::from_points(&pv.nodes.iter().map(|n| n.0).collect::<Vec<_>>());
            let inner = rect.shrink(14.0);
            let s = (inner.width() / b.width().max(1.0)).min(inner.height() / b.height().max(1.0)).min(0.5);
            let at = |q: Pos2| inner.center() + (q - b.center()) * s;
            let wire = ui.visuals().weak_text_color().gamma_multiply(0.6);
            for &(a, c) in &pv.edges {
                p.line_segment([at(pv.nodes[a].0), at(pv.nodes[c].0)], Stroke::new(1.0, wire));
            }
            let r = if pv.nodes.len() > 60 { 2.0 } else { 3.5 };
            for &(q, col) in &pv.nodes {
                p.circle_filled(at(q), r, col);
            }
        }
        ui.add_space(4.0);
        let total: usize = pv.kinds.iter().map(|k| k.1).sum();
        ui.label(format!("{total} node{}", if total == 1 { "" } else { "s" }));
        let parts: Vec<String> = pv
            .kinds
            .iter()
            .take(8)
            .map(|(k, n)| if *n > 1 { format!("{k} ×{n}") } else { k.clone() })
            .collect();
        let more = pv.kinds.len().saturating_sub(8);
        let mut line = parts.join(" · ");
        if more > 0 {
            line.push_str(&format!(" · {more} more"));
        }
        ui.add(egui::Label::new(RichText::new(line).weak().small()).wrap());
        ui.add_space(4.0);
        let mut facts = Vec::new();
        if let Some(f) = pv.fps {
            facts.push(format!("{f} fps"));
        }
        if let Some((a, b)) = pv.range {
            facts.push(format!("frames {a}–{b}"));
        }
        if let Some((w, h)) = pv.canvas {
            facts.push(format!("{w}×{h}"));
        }
        if !facts.is_empty() {
            ui.weak(facts.join(" · "));
        }
        if !pv.plates.is_empty() {
            ui.add_space(4.0);
            ui.label(RichText::new("reads").weak().small());
            for pl in pv.plates.iter().take(5) {
                ui.add(egui::Label::new(RichText::new(pl).small()).truncate());
            }
            if pv.plates.len() > 5 {
                ui.weak(format!("{} more", pv.plates.len() - 5));
            }
        }
    }

    fn bottom_ui(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        let mut act = None;
        match self.mode {
            Mode::Open => {
                let picked = self.sel.as_ref().map(|p| p.display().to_string()).unwrap_or_default();
                ui.add(egui::Label::new(RichText::new(picked).weak()).truncate());
            }
            Mode::Save => {
                ui.label("name");
                let r = ui.add(egui::TextEdit::singleline(&mut self.name).desired_width(260.0).hint_text("comp.wbl"));
                if self.first {
                    r.request_focus();
                }
                if r.changed() {
                    self.confirm = None;
                }
                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    act = Some(self.activate());
                }
                // New folder: a button, then a field in its place.
                match &mut self.new_folder {
                    None => {
                        if ui.button("new folder").clicked() {
                            self.new_folder = Some(String::new());
                            self.focus_folder = true;
                        }
                    }
                    Some(text) => {
                        let f = ui.add(egui::TextEdit::singleline(text).desired_width(140.0).hint_text("folder name"));
                        if std::mem::take(&mut self.focus_folder) {
                            f.request_focus();
                        }
                        if f.lost_focus() {
                            let text = std::mem::take(text);
                            self.new_folder = None;
                            if ui.input(|i| i.key_pressed(egui::Key::Enter)) && !text.trim().is_empty() {
                                match folder_name(&text) {
                                    Err(e) => self.error = Some(e),
                                    Ok(name) => {
                                        let d = self.dir.join(name);
                                        match std::fs::create_dir(&d) {
                                            Ok(()) => self.navigate(d),
                                            Err(e) => self.error = Some(format!("{}: {e}", d.display())),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let (verb, ok) = match self.mode {
                Mode::Open => {
                    let dir = self.sel.as_ref().is_some_and(|p| p.is_dir());
                    (if dir { "open folder" } else { "open" }, self.sel.is_some())
                }
                Mode::Save => ("save", resolve_name(&self.dir, &self.name).is_some()),
            };
            if let Some(path) = self.confirm.clone() {
                let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if ui.button("keep it").clicked() {
                    self.confirm = None;
                }
                if ui.button(RichText::new("replace").color(ui.visuals().warn_fg_color)).clicked() {
                    act = Some(Action::Save(path));
                }
                ui.colored_label(ui.visuals().warn_fg_color, format!("{name} exists — replace it?"));
                return;
            }
            if ui.add_enabled(ok, egui::Button::new(verb)).clicked() {
                act = Some(self.activate());
            }
            if ui.button("cancel").clicked() {
                act = Some(Action::Cancel);
            }
        });
        act
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("wobble-setupbrowser-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_save_name_always_ends_in_wbl() {
        let d = Path::new("/comps");
        assert_eq!(resolve_name(d, "shot10"), Some(d.join("shot10.wbl")));
        assert_eq!(resolve_name(d, "shot10.wbl"), Some(d.join("shot10.wbl")));
        assert_eq!(resolve_name(d, "shot10.WBL"), Some(d.join("shot10.WBL")));
        assert_eq!(resolve_name(d, "v2.json"), Some(d.join("v2.json.wbl")));
        assert_eq!(resolve_name(d, "  "), None);
        assert_eq!(resolve_name(d, "/elsewhere/a"), Some(PathBuf::from("/elsewhere/a.wbl")));
        assert!(resolve_name(d, "~/a").unwrap().starts_with(home()));
    }

    /// Folders first; setups (and old .json only when opening); nothing
    /// else, no dotfiles.
    #[test]
    fn the_list_is_folders_then_setups() {
        let d = scratch("list");
        std::fs::create_dir(d.join("zeta")).unwrap();
        std::fs::create_dir(d.join(".hidden")).unwrap();
        for f in ["b.wbl", "a.wbl", "old.json", "plate.exr", ".x.wbl"] {
            std::fs::write(d.join(f), "{}").unwrap();
        }
        let names = |mode| {
            let mut v = list(&d, mode).unwrap();
            sort_entries(&mut v, Sort::Name, false);
            v.into_iter().map(|e| e.name).collect::<Vec<_>>()
        };
        assert_eq!(names(Mode::Open), ["zeta", "a.wbl", "b.wbl", "old.json"]);
        assert_eq!(names(Mode::Save), ["zeta", "a.wbl", "b.wbl"]);
        let _ = std::fs::remove_dir_all(&d);
    }

    /// Saving over a file asks first; the second press is the yes.
    #[test]
    fn replacing_a_setup_asks_first() {
        let d = scratch("replace");
        std::fs::write(d.join("comp.wbl"), "{}").unwrap();
        let mut b = SetupBrowser::new(Mode::Save, Some(&d), "comp");
        assert_eq!(b.activate(), Action::None);
        assert_eq!(b.confirm, Some(d.join("comp.wbl")));
        assert_eq!(b.activate(), Action::Save(d.join("comp.wbl")));
        // A new name saves at once.
        b.name = "fresh".into();
        b.confirm = None;
        assert_eq!(b.activate(), Action::Save(d.join("fresh.wbl")));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The preview reads the graph off the file: nodes, the wires between
    /// them, what they are and the plates read.
    #[test]
    fn the_preview_reads_the_file() {
        let d = scratch("preview");
        let f = d.join("p.wbl");
        let doc = serde_json::json!({
            "version": 1, "fps": 24.0, "range": [1001, 1100], "canvas": [1920, 1080],
            "graph": {
                "nodes": {
                    "1": {"kind": {"Read": {"path": "/plates/sh010/bg.####.exr"}}, "pos": {"x": 0.0, "y": 0.0}},
                    "2": {"kind": {"NoSuchKind": {}}, "pos": {"x": 0.0, "y": 80.0}},
                    "3": {"kind": "AlsoUnknown", "pos": {"x": 0.0, "y": 160.0}}
                },
                "edges": [[2, 0, 1], [3, 0, 2]]
            }
        });
        std::fs::write(&f, doc.to_string()).unwrap();
        let p = preview_of(&f);
        assert!(p.err.is_none(), "{:?}", p.err);
        assert_eq!(p.nodes.len(), 3);
        assert_eq!(p.edges.len(), 2);
        assert_eq!(p.fps, Some(24.0));
        assert_eq!(p.range, Some((1001, 1100)));
        assert_eq!(p.canvas, Some((1920, 1080)));
        assert_eq!(p.plates, ["bg.####.exr"]);
        let names: Vec<_> = p.kinds.iter().map(|k| k.0.as_str()).collect();
        assert!(names.contains(&"NoSuchKind") && names.contains(&"AlsoUnknown"), "{names:?}");
        assert!(preview_of(&d.join("missing.wbl")).err.is_some());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn ages_read_like_a_person_says_them() {
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_800_000_000);
        let ago = |s: u64| fmt_age(Some(now - std::time::Duration::from_secs(s)), now);
        assert_eq!(ago(5), "just now");
        assert_eq!(ago(600), "10m ago");
        assert_eq!(ago(7200), "2h ago");
        assert_eq!(ago(3 * 86_400), "3d ago");
        assert_eq!(fmt_age(Some(SystemTime::UNIX_EPOCH), now), "1970-01-01");
        assert_eq!(fmt_age(None, now), "—");
    }

    /// Look at it: `cargo test -p app setup_browser_render -- --ignored`,
    /// then `target/setup-browser.png`.
    #[test]
    #[ignore]
    fn setup_browser_render() {
        let d = scratch("render");
        std::fs::create_dir(d.join("old shots")).unwrap();
        let doc = serde_json::json!({
            "version": 1, "fps": 24.0, "range": [1001, 1048], "canvas": [2048, 1080],
            "graph": {
                "nodes": {
                    "1": {"kind": {"Read": {"path": "/plates/sh010/bg.####.exr"}}, "pos": {"x": 0.0, "y": 0.0}},
                    "2": {"kind": {"Read": {"path": "/plates/sh010/fg.####.exr"}}, "pos": {"x": 160.0, "y": 0.0}},
                    "3": {"kind": "Premult", "pos": {"x": 160.0, "y": 90.0}},
                    "4": {"kind": {"Blur": {}}, "pos": {"x": 0.0, "y": 110.0}},
                    "5": {"kind": "Premult", "pos": {"x": 80.0, "y": 200.0}}
                },
                "edges": [[3, 0, 2], [4, 0, 1], [5, 0, 4], [5, 1, 3]]
            }
        });
        for n in ["sh010_comp.wbl", "sh020_comp.wbl", "lookdev.wbl"] {
            std::fs::write(d.join(n), doc.to_string()).unwrap();
        }
        // A real comp for the preview, when this machine has one autosaved.
        let auto = home().join(".wobble/autosave");
        if let Some(real) = std::fs::read_dir(&auto).ok().and_then(|r| {
            r.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "wbl"))
                .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
        }) {
            let _ = std::fs::copy(real, d.join("sh010_comp.wbl"));
        }
        let mut b = SetupBrowser::new(Mode::Open, Some(&d), "");
        b.pick(Some(d.join("sh010_comp.wbl")));
        let recent = [d.join("sh010_comp.wbl"), d.join("gone.wbl")];
        let places = Places { bookmarks: &[], recent: &recent, current: Some(&d.join("sh010_comp.wbl")) };
        crate::uishot::render("setup-browser", 960.0, 600.0, 2.0, 40, |ui, _| {
            let _ = b.show(ui.ctx(), &places);
        });
        let _ = std::fs::remove_dir_all(&d);
    }
}

#[cfg(test)]
mod new_folder_tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("wobble-newfolder-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn key(k: egui::Key, pressed: bool) -> egui::Event {
        egui::Event::Key { key: k, physical_key: None, pressed, repeat: false, modifiers: egui::Modifiers::NONE }
    }

    /// Drive the dialog through egui: open the field, type, press Enter.
    fn type_new_folder(b: &mut SetupBrowser, name: &str) {
        let ctx = egui::Context::default();
        let pass = |b: &mut SetupBrowser, events: Vec<egui::Event>| {
            let places = Places { bookmarks: &[], recent: &[], current: None };
            let raw = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1200.0, 800.0))),
                events,
                ..Default::default()
            };
            let _ = ctx.run_ui(raw, |ui| {
                let _ = b.show(ui.ctx(), &places);
            });
        };
        pass(b, vec![]);
        b.new_folder = Some(String::new());
        b.focus_folder = true;
        pass(b, vec![]);
        pass(b, vec![]);
        pass(b, vec![egui::Event::Text(name.into())]);
        pass(b, vec![key(egui::Key::Enter, true), key(egui::Key::Enter, false)]);
        pass(b, vec![]);
        pass(b, vec![]);
    }

    #[test]
    fn enter_in_the_new_folder_field_makes_the_folder() {
        let d = scratch("enter");
        let mut b = SetupBrowser::new(Mode::Save, Some(&d), "comp");
        type_new_folder(&mut b, "shots");
        assert!(d.join("shots").is_dir(), "no folder; error {:?}, field {:?}", b.error, b.new_folder);
        assert_eq!(b.dir, d.join("shots"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A name Windows would refuse (or silently change) is refused here,
    /// with a reason, on every platform, so a comp's folders travel.
    #[test]
    fn folder_names_windows_would_refuse_are_refused() {
        assert_eq!(folder_name("  shots "), Ok("shots"));
        assert_eq!(folder_name("v001.comp"), Ok("v001.comp"));
        assert_eq!(folder_name("console"), Ok("console"));
        assert_eq!(folder_name("COM10"), Ok("COM10"));
        for bad in ["", " ", ".", "..", "a/b", "a\\b", "V:", "what?", "a*", "<x>", "a|b", "q\"", "trail.", "con", "Nul.txt", "COM1", "lpt9", "tab\tbed"] {
            assert!(folder_name(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn a_bad_folder_name_says_why_and_makes_nothing() {
        let d = scratch("bad");
        let mut b = SetupBrowser::new(Mode::Save, Some(&d), "comp");
        type_new_folder(&mut b, "shot:010");
        assert!(b.error.as_deref().is_some_and(|e| e.contains("cannot contain")), "{:?}", b.error);
        assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0);
        assert_eq!(b.dir, d);
        let _ = std::fs::remove_dir_all(&d);
    }
}
