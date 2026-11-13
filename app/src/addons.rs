//! Addons: Blender-style extensions that go beyond a node.
//!
//! An addon is a folder in a plugin directory (`plugins/`,
//! `~/.wobble/plugins`) holding an `addon.json`. The manifest DECLARES what
//! the addon adds to the app — commands (in the top bar's `addons` menu and
//! the addons window) and panels (floating windows built from a few row
//! kinds) — and may name a process to `run`. When the user enables the
//! addon, the app starts that process in the addon's folder with
//! `WOBBLE_API_PORT` and `WOBBLE_ADDON` set; it connects to the scripting
//! API like any client, waits on `addon_events` for what the user did, and
//! answers with ordinary calls (`node_create`, `addon_panel_set`, ...).
//!
//! Why a process and not an interpreter in the app: scripting is ONE
//! language-neutral protocol (CLAUDE.md, "Direction already decided"), so an
//! addon can be Python, Rust or anything that opens a socket, it can crash
//! without taking the app down, and nothing it does runs per frame — the app
//! draws the declared UI itself, and the process only hears about clicks.
//!
//! Rules:
//! - **Nothing runs until the user enables it.** Enabling starts code on
//!   this machine; the window says so, the list lives in prefs, and a suite
//!   run (default prefs) enables nothing.
//! - **The app owns the UI.** Panels are declared rows the app draws with
//!   the house widgets; the process sets values, never pixels.
//! - Nodes an addon wants ship beside it as `plugin.json` / `tool.json`
//!   folders, which load whether or not the addon is enabled.

use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use web_time::{Duration, Instant};

pub const MANIFEST: &str = "addon.json";
/// Lines of process output kept per addon for the window.
const LOG_LINES: usize = 200;
/// Queued events per addon: a process that stopped listening must not
/// grow the app without bound.
const MAX_EVENTS: usize = 256;

#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub author: String,
    #[serde(default)]
    pub description: String,
    /// The process to start, argv style, run in the addon's folder
    /// (`["python3", "main.py"]`). Empty = a declarative addon.
    #[serde(default)]
    pub run: Vec<String>,
    #[serde(default)]
    pub commands: Vec<CommandDef>,
    #[serde(default)]
    pub panels: Vec<PanelDef>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CommandDef {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub tip: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PanelDef {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub rows: Vec<RowDef>,
}

/// One row of a panel. The id is what events and `addon_panel_set` name.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum RowDef {
    Button { id: String, label: String },
    Slider {
        id: String,
        label: String,
        #[serde(default)]
        min: f32,
        #[serde(default = "one")]
        max: f32,
        #[serde(default)]
        default: f32,
        #[serde(default)]
        integer: bool,
    },
    Toggle {
        id: String,
        label: String,
        #[serde(default)]
        default: bool,
    },
    /// A line of text the process writes.
    Text {
        id: String,
        #[serde(default)]
        text: String,
    },
    /// A text field: an event when the user commits it (Enter / focus out).
    Field {
        id: String,
        label: String,
        #[serde(default)]
        default: String,
    },
    /// A dropdown: an event with the picked option. The process may replace
    /// the options at any time (`addon_panel_set` with `{"options": [..],
    /// "value": ..}`), since what there is to pick from is usually only known
    /// at run time.
    Choice {
        id: String,
        label: String,
        #[serde(default)]
        options: Vec<String>,
        #[serde(default)]
        default: String,
    },
}

fn one() -> f32 {
    1.0
}

impl RowDef {
    pub fn id(&self) -> &str {
        match self {
            RowDef::Button { id, .. }
            | RowDef::Slider { id, .. }
            | RowDef::Toggle { id, .. }
            | RowDef::Text { id, .. }
            | RowDef::Field { id, .. }
            | RowDef::Choice { id, .. } => id,
        }
    }

    fn default_value(&self) -> Value {
        match self {
            RowDef::Button { label, .. } => json!(label),
            RowDef::Slider { default, .. } => json!(default),
            RowDef::Toggle { default, .. } => json!(default),
            RowDef::Text { text, .. } => json!(text),
            RowDef::Field { default, .. } | RowDef::Choice { default, .. } => json!(default),
        }
    }
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self, String> {
        let m: Manifest = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        if !ok(&m.id) {
            return Err(format!("id '{}' must be letters, digits, _ or -", m.id));
        }
        let mut seen = HashSet::new();
        for c in &m.commands {
            if !ok(&c.id) || !seen.insert(format!("c/{}", c.id)) {
                return Err(format!("command id '{}' is empty, malformed or repeated", c.id));
            }
        }
        for p in &m.panels {
            if !ok(&p.id) || !seen.insert(format!("p/{}", p.id)) {
                return Err(format!("panel id '{}' is empty, malformed or repeated", p.id));
            }
            let mut rows = HashSet::new();
            for r in &p.rows {
                if !rows.insert(r.id()) {
                    return Err(format!("panel '{}' repeats row '{}'", p.id, r.id()));
                }
            }
        }
        Ok(m)
    }
}

/// A running addon process and the output it has printed.
struct Proc {
    child: Child,
    log: Arc<Mutex<VecDeque<String>>>,
}

pub struct Addon {
    pub dir: PathBuf,
    pub m: Manifest,
    pub enabled: bool,
    proc: Option<Proc>,
    /// Output of the last process, kept after it exits so a crash can be read.
    log: Arc<Mutex<VecDeque<String>>>,
    /// Why the process is not running, when it should be.
    pub exit: Option<String>,
    /// The process's own one-line status (`addon_status`).
    pub status: String,
    events: VecDeque<Value>,
    /// Row values, keyed `panel/row`.
    values: HashMap<String, Value>,
    /// A choice row's options, keyed `panel/row`.
    options: HashMap<String, Vec<String>>,
    pub open_panels: HashSet<String>,
}

impl Addon {
    fn new(dir: PathBuf, m: Manifest) -> Self {
        let mut values = HashMap::new();
        let mut options = HashMap::new();
        for p in &m.panels {
            for r in &p.rows {
                values.insert(format!("{}/{}", p.id, r.id()), r.default_value());
                if let RowDef::Choice { options: o, .. } = r {
                    options.insert(format!("{}/{}", p.id, r.id()), o.clone());
                }
            }
        }
        Self {
            dir,
            m,
            enabled: false,
            proc: None,
            log: Default::default(),
            exit: None,
            status: String::new(),
            events: VecDeque::new(),
            values,
            options,
            open_panels: HashSet::new(),
        }
    }

    pub fn running(&self) -> bool {
        self.proc.is_some()
    }

    fn push(&mut self, ev: Value) {
        // A slider drag is a stream of values; only the latest matters, so
        // one not yet collected is replaced rather than queued behind.
        if ev["type"] == "panel" {
            if let Some(last) = self.events.back_mut() {
                if last["type"] == "panel" && last["panel"] == ev["panel"] && last["row"] == ev["row"] && !ev["value"].is_null() {
                    *last = ev;
                    return;
                }
            }
        }
        if self.events.len() >= MAX_EVENTS {
            self.events.pop_front();
        }
        self.events.push_back(ev);
    }

    fn start(&mut self, port: u16) {
        self.stop();
        let Some((exe, args)) = self.m.run.split_first() else { return };
        let mut cmd = Command::new(exe);
        cmd.args(args)
            .current_dir(&self.dir)
            .env("WOBBLE_API_PORT", port.to_string())
            .env("WOBBLE_ADDON", &self.m.id)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // The in-repo Python client, so an addon under development finds
        // `import wobble` without a pip install. An installed client wins.
        if let Ok(py) = std::fs::canonicalize("clients/python") {
            let mut paths = vec![py];
            if let Some(old) = std::env::var_os("PYTHONPATH") {
                paths.extend(std::env::split_paths(&old));
            }
            if let Ok(joined) = std::env::join_paths(paths) {
                cmd.env("PYTHONPATH", joined);
            }
        }
        let log: Arc<Mutex<VecDeque<String>>> = Default::default();
        match cmd.spawn() {
            Ok(mut child) => {
                let out = child.stdout.take();
                let err = child.stderr.take();
                if let Some(o) = out {
                    pipe(o, log.clone(), "");
                }
                if let Some(e) = err {
                    pipe(e, log.clone(), "! ");
                }
                self.exit = None;
                self.log = log.clone();
                self.proc = Some(Proc { child, log });
            }
            Err(e) => self.exit = Some(format!("could not start {exe}: {e}")),
        }
    }

    fn stop(&mut self) {
        if let Some(mut p) = self.proc.take() {
            let _ = p.child.kill();
            let _ = p.child.wait();
            self.log = p.log;
        }
    }

    fn reap(&mut self) {
        let Some(p) = &mut self.proc else { return };
        if let Ok(Some(st)) = p.child.try_wait() {
            self.exit = Some(format!("process exited ({st})"));
            let p = self.proc.take().unwrap();
            self.log = p.log;
        }
    }

    pub fn log_lines(&self) -> Vec<String> {
        self.log.lock().map(|l| l.iter().cloned().collect()).unwrap_or_default()
    }
}

fn pipe(r: impl std::io::Read + Send + 'static, log: Arc<Mutex<VecDeque<String>>>, prefix: &'static str) {
    let _ = std::thread::Builder::new().name("addon-log".into()).spawn(move || {
        for line in BufReader::new(r).lines().map_while(Result::ok) {
            if let Ok(mut l) = log.lock() {
                if l.len() >= LOG_LINES {
                    l.pop_front();
                }
                l.push_back(format!("{prefix}{line}"));
            }
        }
    });
}

/// Every addon found, enabled or not.
#[derive(Default)]
pub struct Addons {
    pub list: Vec<Addon>,
    pub errors: Vec<String>,
    pub window_open: bool,
    last_reap: Option<Instant>,
}

impl Drop for Addons {
    fn drop(&mut self) {
        for a in &mut self.list {
            a.stop();
        }
    }
}

/// Scan `dirs` for `<name>/addon.json`. The first of two addons with the
/// same id wins (the repo's `plugins/` is scanned before `~/.wobble`).
pub fn scan(dirs: &[PathBuf]) -> (Vec<(PathBuf, Manifest)>, Vec<String>) {
    let mut found: Vec<(PathBuf, Manifest)> = Vec::new();
    let mut errors = Vec::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(d) else { continue };
        let mut subs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.join(MANIFEST).is_file()).collect();
        subs.sort();
        for sub in subs {
            let path = sub.join(MANIFEST);
            match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| Manifest::parse(&t)) {
                Ok(m) if found.iter().any(|(_, f)| f.id == m.id) => {
                    errors.push(format!("{}: id '{}' already taken", path.display(), m.id))
                }
                Ok(m) => found.push((sub, m)),
                Err(e) => errors.push(format!("{}: {e}", path.display())),
            }
        }
    }
    (found, errors)
}

/// Menu ids are `&'static str` (the tilt menus match on them); addon ids
/// are data. Interned once each, so a reload does not leak another copy.
fn intern(s: String) -> &'static str {
    static SEEN: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap();
    let set = g.get_or_insert_with(HashSet::new);
    if let Some(k) = set.get(s.as_str()) {
        return k;
    }
    let k: &'static str = Box::leak(s.into_boxed_str());
    set.insert(k);
    k
}

/// What a tilt-menu item id means, for the ids this module mints.
pub enum MenuHit {
    Command(String, String),
    Panel(String, String),
}

pub fn command_item(addon: &str, cmd: &str) -> &'static str {
    intern(format!("addon-cmd\u{1f}{addon}\u{1f}{cmd}"))
}

pub fn panel_item(addon: &str, panel: &str) -> &'static str {
    intern(format!("addon-panel\u{1f}{addon}\u{1f}{panel}"))
}

pub fn menu_hit(id: &str) -> Option<MenuHit> {
    let mut it = id.split('\u{1f}');
    let (kind, a, b) = (it.next()?, it.next()?.to_string(), it.next()?.to_string());
    match kind {
        "addon-cmd" => Some(MenuHit::Command(a, b)),
        "addon-panel" => Some(MenuHit::Panel(a, b)),
        _ => None,
    }
}

impl Addons {
    /// Rescan, keeping running processes of addons that are still there
    /// and unchanged in id. `enabled` is the prefs list.
    pub fn load(&mut self, dirs: &[PathBuf], enabled: &[String]) {
        let (found, errors) = scan(dirs);
        let mut old: HashMap<String, Addon> = self.list.drain(..).map(|a| (a.m.id.clone(), a)).collect();
        for (dir, m) in found {
            let mut a = Addon::new(dir, m);
            if let Some(mut prev) = old.remove(&a.m.id) {
                // Same addon: carry the live state across the rescan.
                a.proc = prev.proc.take();
                a.log = prev.log.clone();
                a.exit = prev.exit.take();
                a.status = std::mem::take(&mut prev.status);
                a.events = std::mem::take(&mut prev.events);
                a.open_panels = std::mem::take(&mut prev.open_panels);
                for (k, v) in prev.values.drain() {
                    if a.values.contains_key(&k) {
                        a.values.insert(k, v);
                    }
                }
            }
            a.enabled = enabled.contains(&a.m.id);
            self.list.push(a);
        }
        // `old` now holds addons whose folder went away; dropping them
        // kills their processes.
        for (_, mut a) in old {
            a.stop();
        }
        self.errors = errors;
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Addon> {
        self.list.iter_mut().find(|a| a.m.id == id)
    }

    /// Start every enabled addon that should be running and is not (after
    /// a load, or once the API is up). `port` is None while there is no API
    /// to hand them: they wait.
    pub fn ensure_running(&mut self, port: Option<u16>) {
        let Some(port) = port else { return };
        for a in &mut self.list {
            if a.enabled && !a.m.run.is_empty() && a.proc.is_none() && a.exit.is_none() {
                a.start(port);
            }
        }
    }

    /// Whether any enabled addon wants a process — i.e. whether the app
    /// has to open the API for it.
    pub fn wants_api(&self) -> bool {
        self.list.iter().any(|a| a.enabled && !a.m.run.is_empty())
    }

    pub fn set_enabled(&mut self, id: &str, on: bool, port: Option<u16>) {
        let Some(a) = self.get_mut(id) else { return };
        a.enabled = on;
        a.exit = None;
        if on {
            if let Some(p) = port {
                a.start(p);
            }
        } else {
            a.stop();
            a.open_panels.clear();
            a.events.clear();
        }
    }

    pub fn restart(&mut self, id: &str, port: Option<u16>) {
        if let (Some(a), Some(p)) = (self.get_mut(id), port) {
            a.start(p);
        }
    }

    /// Check processes for an exit, at most twice a second.
    pub fn reap(&mut self) {
        let now = Instant::now();
        if self.last_reap.is_some_and(|t| now - t < Duration::from_millis(500)) {
            return;
        }
        self.last_reap = Some(now);
        for a in &mut self.list {
            a.reap();
        }
    }

    pub fn command(&mut self, addon: &str, cmd: &str) {
        if let Some(a) = self.get_mut(addon).filter(|a| a.enabled) {
            a.push(json!({ "type": "command", "command": cmd }));
        }
    }

    /// For `addon_events`: everything queued, oldest first.
    pub fn take_events(&mut self, addon: &str) -> Vec<Value> {
        self.get_mut(addon).map(|a| a.events.drain(..).collect()).unwrap_or_default()
    }

    pub fn panel_set(&mut self, addon: &str, panel: &str, values: &Map<String, Value>, open: Option<bool>) -> Result<(), String> {
        let a = self.get_mut(addon).ok_or_else(|| format!("no addon '{addon}'"))?;
        let def = a.m.panels.iter().find(|p| p.id == panel).ok_or_else(|| format!("addon '{addon}' has no panel '{panel}'"))?;
        for k in values.keys() {
            if !def.rows.iter().any(|r| r.id() == k) {
                return Err(format!("panel '{panel}' has no row '{k}'"));
            }
        }
        for (k, v) in values {
            let key = format!("{panel}/{k}");
            let choice = def.rows.iter().any(|r| r.id() == k && matches!(r, RowDef::Choice { .. }));
            match v {
                // A choice row takes its options and its pick in one object.
                Value::Object(o) if choice => {
                    if let Some(opts) = o.get("options") {
                        let list = opts.as_array().ok_or_else(|| format!("row '{k}': options must be a list of strings"))?;
                        let list: Vec<String> = list.iter().map(|x| x.as_str().map(str::to_string).unwrap_or_else(|| x.to_string())).collect();
                        a.options.insert(key.clone(), list);
                    }
                    if let Some(pick) = o.get("value") {
                        a.values.insert(key, pick.clone());
                    }
                }
                _ => {
                    a.values.insert(key, v.clone());
                }
            }
        }
        match open {
            Some(true) => {
                a.open_panels.insert(panel.to_string());
            }
            Some(false) => {
                a.open_panels.remove(panel);
            }
            None => {}
        }
        Ok(())
    }

    pub fn set_status(&mut self, addon: &str, text: &str) -> Result<(), String> {
        let a = self.get_mut(addon).ok_or_else(|| format!("no addon '{addon}'"))?;
        a.status = text.to_string();
        Ok(())
    }

    /// Draw every open panel as a floating window.
    pub fn panels_ui(&mut self, ctx: &egui::Context) {
        for a in &mut self.list {
            if !a.enabled {
                continue;
            }
            for p in a.m.panels.clone() {
                if !a.open_panels.contains(&p.id) {
                    continue;
                }
                let mut open = true;
                egui::Window::new(format!("{} — {}", a.m.label, p.label))
                    .id(egui::Id::new(("addon-panel", &a.m.id, &p.id)))
                    .open(&mut open)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        panel_rows(ui, a, &p);
                        if !a.status.is_empty() {
                            ui.separator();
                            ui.weak(&a.status);
                        }
                        if !a.m.run.is_empty() && !a.running() {
                            ui.colored_label(crate::theme::warn(), a.exit.as_deref().unwrap_or("not running"));
                        }
                    });
                if !open {
                    a.open_panels.remove(&p.id);
                }
            }
        }
    }
}

fn panel_rows(ui: &mut egui::Ui, a: &mut Addon, p: &PanelDef) {
    for r in &p.rows {
        let key = format!("{}/{}", p.id, r.id());
        let cur = a.values.get(&key).cloned().unwrap_or(Value::Null);
        let changed: Option<Value> = match r {
            RowDef::Button { id, .. } => {
                let label = cur.as_str().unwrap_or(id).to_string();
                ui.button(label).clicked().then_some(Value::Null)
            }
            RowDef::Slider { label, min, max, integer, .. } => {
                let mut v = cur.as_f64().unwrap_or(0.0) as f32;
                let r = crate::widgets::plain_slider(ui, label, &mut v, *min, *max, *integer);
                r.changed().then(|| json!(v))
            }
            RowDef::Toggle { label, .. } => {
                let mut v = cur.as_bool().unwrap_or(false);
                ui.checkbox(&mut v, label.as_str()).changed().then(|| json!(v))
            }
            RowDef::Text { .. } => {
                ui.label(cur.as_str().unwrap_or_default());
                None
            }
            RowDef::Field { label, .. } => {
                let mut v = cur.as_str().unwrap_or_default().to_string();
                let resp = ui
                    .horizontal(|ui| {
                        ui.add_sized([crate::widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new(label).small()));
                        ui.text_edit_singleline(&mut v)
                    })
                    .inner;
                if resp.changed() {
                    a.values.insert(key.clone(), json!(v));
                }
                resp.lost_focus().then(|| json!(v))
            }
            RowDef::Choice { label, .. } => {
                let opts = a.options.get(&key).cloned().unwrap_or_default();
                let pick = cur.as_str().unwrap_or_default().to_string();
                let mut picked = None;
                ui.horizontal(|ui| {
                    ui.add_sized([crate::widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new(label).small()));
                    let shown = if pick.is_empty() { "—" } else { pick.as_str() };
                    egui::ComboBox::from_id_salt(("addon-choice", &key)).selected_text(shown).width(ui.available_width()).show_ui(ui, |ui| {
                        for o in &opts {
                            if ui.selectable_label(*o == pick, o.as_str()).clicked() && *o != pick {
                                picked = Some(json!(o));
                            }
                        }
                    });
                });
                picked
            }
        };
        if let Some(v) = changed {
            if !v.is_null() {
                a.values.insert(key, v.clone());
            }
            a.push(json!({ "type": "panel", "panel": p.id, "row": r.id(), "value": v }));
        }
    }
}

/// Tilt-menu items for the enabled addons: one submenu each, commands then
/// panel toggles.
pub fn menu_items(addons: &Addons) -> Vec<crate::tiltmenu::Item> {
    use crate::tiltmenu::Item;
    let mut items = Vec::new();
    for a in addons.list.iter().filter(|a| a.enabled) {
        let mut sub: Vec<Item> = a.m.commands.iter().map(|c| Item::action(command_item(&a.m.id, &c.id), c.label.clone())).collect();
        for p in &a.m.panels {
            sub.push(Item::check(panel_item(&a.m.id, &p.id), format!("{} panel", p.label), a.open_panels.contains(&p.id)));
        }
        if !sub.is_empty() {
            items.push(Item::sub(a.m.label.clone(), sub));
        }
    }
    items
}

/// The folders addons are found in: the same as plugins and tools.
pub fn dirs() -> Vec<PathBuf> {
    let mut d = vec![PathBuf::from("plugins")];
    if let Some(h) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        d.push(Path::new(&h).join(".wobble/plugins"));
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_example_addon_parses() {
        let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../plugins/hello_addon/addon.json")).unwrap();
        let m = Manifest::parse(&t).unwrap();
        assert_eq!(m.id, "hello_addon");
        assert!(!m.commands.is_empty() && !m.panels.is_empty() && !m.run.is_empty());
    }

    #[test]
    fn bad_ids_and_repeats_are_refused() {
        assert!(Manifest::parse(r#"{"id":"a b","label":"x"}"#).is_err());
        assert!(Manifest::parse(r#"{"id":"a","label":"x","commands":[{"id":"c","label":"c"},{"id":"c","label":"d"}]}"#).is_err());
        assert!(Manifest::parse(r#"{"id":"a","label":"x","panels":[{"id":"p","label":"p","rows":[{"kind":"button","id":"b","label":"b"},{"kind":"text","id":"b"}]}]}"#).is_err());
        assert!(Manifest::parse(r#"{"id":"a","label":"x"}"#).is_ok());
    }

    #[test]
    fn menu_ids_round_trip_and_intern() {
        let id = command_item("hello", "go");
        assert!(std::ptr::eq(id, command_item("hello", "go")));
        assert!(matches!(menu_hit(id), Some(MenuHit::Command(a, c)) if a == "hello" && c == "go"));
        assert!(matches!(menu_hit(panel_item("hello", "main")), Some(MenuHit::Panel(a, p)) if a == "hello" && p == "main"));
        assert!(menu_hit("colours").is_none());
    }

    fn addon() -> Addons {
        let m = Manifest::parse(
            r#"{"id":"a","label":"A","commands":[{"id":"go","label":"go"}],
                "panels":[{"id":"p","label":"P","rows":[{"kind":"slider","id":"s","label":"s","default":0.25}]}]}"#,
        )
        .unwrap();
        let mut a = Addon::new(PathBuf::new(), m);
        a.enabled = true;
        let mut ad = Addons::default();
        ad.list.push(a);
        ad
    }

    #[test]
    fn events_queue_and_a_drag_coalesces() {
        let mut ad = addon();
        ad.command("a", "go");
        let a = ad.get_mut("a").unwrap();
        for v in [0.3, 0.4, 0.5] {
            a.push(json!({ "type": "panel", "panel": "p", "row": "s", "value": v }));
        }
        let ev = ad.take_events("a");
        assert_eq!(ev.len(), 2, "{ev:?}");
        assert_eq!(ev[1]["value"], 0.5);
        assert!(ad.take_events("a").is_empty());
    }

    #[test]
    fn a_disabled_addon_hears_nothing() {
        let mut ad = addon();
        ad.set_enabled("a", false, None);
        ad.command("a", "go");
        assert!(ad.take_events("a").is_empty());
    }

    #[test]
    fn panel_set_checks_rows_and_opens() {
        let mut ad = addon();
        let mut v = Map::new();
        v.insert("s".into(), json!(0.9));
        ad.panel_set("a", "p", &v, Some(true)).unwrap();
        let a = ad.get_mut("a").unwrap();
        assert_eq!(a.values["p/s"], 0.9);
        assert!(a.open_panels.contains("p"));
        v.insert("nope".into(), json!(1));
        assert!(ad.panel_set("a", "p", &v, None).is_err());
        assert!(ad.panel_set("a", "q", &Map::new(), None).is_err());
    }

    #[test]
    fn a_choice_takes_its_options_and_pick_together() {
        let m = Manifest::parse(
            r#"{"id":"a","label":"A","panels":[{"id":"p","label":"P","rows":[
                {"kind":"choice","id":"c","label":"c","options":["x","y"],"default":"x"}]}]}"#,
        )
        .unwrap();
        let mut ad = Addons::default();
        ad.list.push(Addon::new(PathBuf::new(), m));
        assert_eq!(ad.get_mut("a").unwrap().options["p/c"], vec!["x", "y"]);
        let mut v = Map::new();
        v.insert("c".into(), json!({ "options": ["m", "n", "o"], "value": "n" }));
        ad.panel_set("a", "p", &v, None).unwrap();
        let a = ad.get_mut("a").unwrap();
        assert_eq!(a.options["p/c"], vec!["m", "n", "o"]);
        assert_eq!(a.values["p/c"], "n");
        // A plain string still just picks.
        v.insert("c".into(), json!("o"));
        ad.panel_set("a", "p", &v, None).unwrap();
        assert_eq!(ad.get_mut("a").unwrap().values["p/c"], "o");
        let mut bad = Map::new();
        bad.insert("c".into(), json!({ "options": "nope" }));
        assert!(ad.panel_set("a", "p", &bad, None).is_err());
    }

    #[test]
    fn the_organisicator_addon_parses() {
        let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../plugins/organisicator/addon.json")).unwrap();
        let m = Manifest::parse(&t).unwrap();
        assert_eq!(m.id, "organisicator");
    }
}

impl crate::Gfx {
    /// Once per UI pass: notice exited processes, open the API if an
    /// enabled addon needs one, start what should be running, draw the
    /// addons window and the open panels.
    pub(crate) fn addons_frame(&mut self, ctx: &egui::Context) {
        self.addons.reap();
        if self.addons.wants_api() && self.api.is_none() {
            // Port 0: a free one. An addon learns it from WOBBLE_API_PORT,
            // so the fixed 7979 stays free for `--api` in another instance.
            self.api = crate::apiserve::ApiServe::start(0);
        }
        let port = self.api.as_ref().map(|a| a.port());
        self.addons.ensure_running(port);
        if self.addons.window_open {
            self.addons_window(ctx, port);
        }
        self.addons.panels_ui(ctx);
    }

    pub(crate) fn addons_reload(&mut self) {
        let enabled = self.prefs.addons_enabled.clone();
        self.addons.load(&dirs(), &enabled);
    }

    fn addons_window(&mut self, ctx: &egui::Context, port: Option<u16>) {
        let mut open = true;
        let mut toggles: Vec<(String, bool)> = Vec::new();
        let mut restart: Option<String> = None;
        let mut reload = false;
        egui::Window::new("addons")
            .open(&mut open)
            .resizable(true)
            .default_width(460.0)
            .show(ctx, |ui| {
                ui.weak(
                    "An addon is a folder with an addon.json in plugins/ or ~/.wobble/plugins. \
                     Enabling one that runs a process starts that program on this machine.",
                );
                ui.horizontal(|ui| {
                    if ui.button("rescan").clicked() {
                        reload = true;
                    }
                    if let Some(p) = port {
                        ui.weak(format!("API on 127.0.0.1:{p}"));
                    }
                });
                ui.separator();
                if self.addons.list.is_empty() {
                    ui.weak("no addons found");
                }
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for a in &mut self.addons.list {
                        ui.horizontal(|ui| {
                            let mut on = a.enabled;
                            if ui.checkbox(&mut on, egui::RichText::new(&a.m.label).strong()).changed() {
                                toggles.push((a.m.id.clone(), on));
                            }
                            if !a.m.version.is_empty() {
                                ui.weak(&a.m.version);
                            }
                            if !a.m.author.is_empty() {
                                ui.weak(format!("· {}", a.m.author));
                            }
                        });
                        ui.indent(("addon", a.m.id.clone()), |ui| {
                            if !a.m.description.is_empty() {
                                ui.label(&a.m.description);
                            }
                            ui.weak(a.dir.display().to_string());
                            if !a.m.run.is_empty() {
                                ui.weak(format!("runs: {}", a.m.run.join(" ")));
                            }
                            if a.enabled {
                                if a.running() {
                                    ui.label("running");
                                } else if let Some(e) = &a.exit {
                                    ui.horizontal(|ui| {
                                        ui.colored_label(crate::theme::warn(), e);
                                        if ui.button("restart").clicked() {
                                            restart = Some(a.m.id.clone());
                                        }
                                    });
                                }
                                if !a.status.is_empty() {
                                    ui.weak(&a.status);
                                }
                                ui.horizontal_wrapped(|ui| {
                                    for c in a.m.commands.clone() {
                                        let b = ui.button(&c.label);
                                        let b = if c.tip.is_empty() { b } else { b.on_hover_text(&c.tip) };
                                        if b.clicked() {
                                            a.push(json!({ "type": "command", "command": c.id }));
                                        }
                                    }
                                    for p in &a.m.panels {
                                        let mut o = a.open_panels.contains(&p.id);
                                        if ui.toggle_value(&mut o, format!("{} panel", p.label)).changed() {
                                            if o {
                                                a.open_panels.insert(p.id.clone());
                                            } else {
                                                a.open_panels.remove(&p.id);
                                            }
                                        }
                                    }
                                });
                                let log = a.log_lines();
                                if !log.is_empty() {
                                    egui::CollapsingHeader::new(format!("output ({})", log.len()))
                                        .id_salt(("addon-log", &a.m.id))
                                        .show(ui, |ui| {
                                            egui::ScrollArea::vertical().max_height(160.0).stick_to_bottom(true).show(ui, |ui| {
                                                for l in &log {
                                                    ui.monospace(l);
                                                }
                                            });
                                        });
                                }
                            }
                        });
                        ui.add_space(6.0);
                    }
                    for e in &self.addons.errors {
                        ui.colored_label(crate::theme::warn(), e);
                    }
                });
            });
        self.addons.window_open = open;
        for (id, on) in toggles {
            self.addons.set_enabled(&id, on, port);
            let list = &mut self.prefs.addons_enabled;
            list.retain(|x| *x != id);
            if on {
                list.push(id);
            }
            crate::save_prefs(&self.prefs);
        }
        if let Some(id) = restart {
            self.addons.restart(&id, port);
        }
        if reload {
            self.addons_reload();
        }
    }
}
