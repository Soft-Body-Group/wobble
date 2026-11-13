//! wobble-colors: design Wobble's colour scheme by picking a handful of
//! shared colours.
//!
//! The designer sets only the colours Wobble repeats everywhere ([`CORE`]):
//! background, text, lines, accent and so on. Every other colour is a
//! variation Wobble's theme derives from one of them (its `shade` / `like`
//! links), so it moves along without a decision. Mock layouts of Wobble's
//! screens show the result; clicking anything picks the shared colour it
//! comes from.
//!
//! Nodes are the one place that goes deeper: every node type is a shade of
//! its [`FAMILIES`] colour, and a double-click on a node tunes that node
//! alone. Their icons are Wobble's own shapes ([`glyphs`], a copy Wobble's
//! tests hold to the graph).
//!
//! The theme table and picker are Wobble's own files, shared by #[path], so
//! slot ids, links and prefs format cannot drift. "save to Wobble" writes the
//! overrides into `~/.wobble/prefs.json` (the `theme` key, plus the picker
//! swatches) and touches nothing else there.

#[allow(dead_code)]
#[path = "../../app/src/theme.rs"]
mod theme;
#[allow(dead_code)]
#[path = "../../app/src/colorpicker.rs"]
mod colorpicker;
#[allow(dead_code)]
#[path = "../../app/src/numfield.rs"]
mod numfield;

mod canvas;
mod glyphs;
mod scenes;
mod shot;

use colorpicker::{dec1, enc1, PickSpec};
use egui::{Color32, RichText, Sense, Vec2};
use scenes::Scene;
use std::collections::BTreeMap;
use theme::{Slot, DEFS};

type Overrides = BTreeMap<String, String>;

/// The colours a designer picks. Everything else follows one of these.
const CORE: &[(Slot, &str, &str)] = &[
    (Slot::bg, "Background", "panels, windows, fields, canvases"),
    (Slot::text, "Text", "labels, values, node names"),
    (Slot::border, "Lines", "borders, separators, outlines"),
    (Slot::button, "Buttons", "buttons, sliders, checkboxes"),
    (Slot::accent, "Accent", "selected items, text selection"),
    (Slot::highlight, "Highlight", "on states, animated values, cables"),
    (Slot::selection, "Selection", "selected outlines, the playhead"),
    (Slot::wire, "Wires", "node graph connections"),
    (Slot::handle, "Handles", "knobs, points, gizmo handles"),
    (Slot::ok, "Good", "ok, cached"),
    (Slot::warn, "Warning", "warnings, selected keys"),
    (Slot::error, "Error", "errors, keyframes, timeline playheads"),
];

/// The node families. Every node type is a shade of one, so moving a
/// family recolours all its nodes and keeps them apart.
const FAMILIES: &[(Slot, &str)] = &[
    (Slot::family_source, "Sources"),
    (Slot::family_colour, "Colour"),
    (Slot::family_filter, "Filters"),
    (Slot::family_transform, "Transforms"),
    (Slot::family_matte, "Mattes + keys"),
    (Slot::family_channels, "Channels"),
    (Slot::family_merge, "Merges"),
    (Slot::family_time, "Time"),
    (Slot::family_track, "Tracking"),
    (Slot::family_other, "Output + other"),
];

/// What every node is drawn with besides its family colour. Some icons are
/// line work in the outline colour alone (Shader, LoopIn), so these decide
/// whether those read at all.
const NODE_LINES: &[(Slot, &str, &str)] = &[
    (Slot::node_outline, "Node outlines", "outlines, and icons drawn as lines"),
    (Slot::node_ink, "Node ink", "the dark marks on a node's body"),
];

fn family_name(s: Slot) -> &'static str {
    FAMILIES.iter().find(|f| f.0 == s).map_or("", |f| f.1)
}

/// The node wearing `s`, when a node does.
fn node_of(s: Slot) -> Option<&'static glyphs::Node> {
    glyphs::NODES.iter().find(|n| n.slot == s)
}

/// Does `s` take its colour from `from` — is it `from`, or linked up to it?
fn follows(s: Slot, from: Slot) -> bool {
    let mut cur = s;
    loop {
        if cur == from {
            return true;
        }
        match DEFS[cur as usize].base.parent() {
            Some(p) => cur = p,
            None => return false,
        }
    }
}

/// The family a node's colour follows; `None` for nodes that borrow a
/// colour from elsewhere (ToData wears the data wires').
fn family_of(s: Slot) -> Option<Slot> {
    FAMILIES.iter().map(|f| f.0).find(|&f| follows(s, f))
}

/// The colour a click on `s` picks: the shared or family colour up its
/// links, else a node's own colour. `None`: not the designer's to set.
fn pick_of(s: Slot) -> Option<Slot> {
    let mut cur = s;
    loop {
        if CORE.iter().any(|c| c.0 == cur)
            || FAMILIES.iter().any(|f| f.0 == cur)
            || NODE_LINES.iter().any(|l| l.0 == cur)
        {
            return Some(cur);
        }
        match DEFS[cur as usize].base.parent() {
            Some(p) => cur = p,
            None => return node_of(s).map(|_| s),
        }
    }
}

fn name_of(s: Slot) -> &'static str {
    CORE.iter()
        .find(|c| c.0 == s)
        .map(|c| c.1)
        .or_else(|| FAMILIES.iter().find(|f| f.0 == s).map(|f| f.1))
        .or_else(|| NODE_LINES.iter().find(|l| l.0 == s).map(|l| l.1))
        .or_else(|| node_of(s).map(|n| n.name))
        .unwrap_or(DEFS[s as usize].label)
}

/// A few member names, for a family's row.
fn names_hint(members: &[&glyphs::Node]) -> String {
    let mut out = String::new();
    for (i, n) in members.iter().enumerate() {
        if out.len() + n.name.len() > 34 {
            out.push_str(&format!(" +{}", members.len() - i));
            break;
        }
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(n.name);
    }
    out
}

fn main() -> eframe::Result {
    env_logger::init();
    // `--shot out.png`: render one frame with no window and quit.
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--shot") {
        let Some(path) = args.get(i + 1) else {
            eprintln!("--shot needs a file name");
            std::process::exit(2);
        };
        // `--click x,y` / `--dclick x,y`: a click, or two, at that point
        // (window points) before the shot, to check what they select.
        let point = |flag: &str| {
            let i = args.iter().position(|a| a == flag)?;
            let (x, y) = args.get(i + 1)?.split_once(',')?;
            Some(egui::pos2(x.parse().ok()?, y.parse().ok()?))
        };
        let (click, dclick) = (point("--click"), point("--dclick"));
        let events = |f: usize| {
            let mut ev = Vec::new();
            if let Some(p) = click {
                ev.extend(shot::click(p, 4, f));
            }
            if let Some(p) = dclick {
                ev.extend(shot::click(p, 4, f));
                ev.extend(shot::click(p, 8, f));
            }
            ev
        };
        let ctx = egui::Context::default();
        let mut app = App::new(&ctx);
        if let Err(e) = shot::render(path.as_ref(), &ctx, (1560.0, 920.0), 1.0, 60, events, |ui| app.frame(ui)) {
            eprintln!("--shot: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("wobble colours")
            .with_inner_size([1560.0, 920.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native("wobble colours", options, Box::new(|cc| Ok(Box::new(App::new(&cc.egui_ctx)))))
}

fn prefs_path() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".wobble/prefs.json"))
}

fn read_prefs() -> serde_json::Value {
    prefs_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}))
}

/// A theme out of any JSON we might be handed: a prefs file, an exported
/// theme (`{"theme": {...}}`), or a bare `{id: hex}` map.
fn theme_of(v: &serde_json::Value) -> Option<Overrides> {
    let m = v.get("theme").unwrap_or(v).as_object()?;
    Some(m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect())
}

fn swatches_of(v: &serde_json::Value) -> Option<Vec<[f32; 3]>> {
    serde_json::from_value(v.get("picker_swatches")?.clone()).ok()
}

fn wobble_running() -> bool {
    std::process::Command::new("pgrep")
        .args(["-x", "wobble"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

struct App {
    scene: Scene,
    /// The core colour being edited.
    sel: Option<Slot>,
    /// A core colour hovered in the list: lit on the canvas this frame.
    panel_hover: Option<Slot>,
    /// The selection came from the canvas: scroll the list to its row, for
    /// this many frames (a picker opening under it settles its height on the
    /// frame after).
    reveal: u8,
    view: colorpicker::View,
    undo: Vec<Overrides>,
    redo: Vec<Overrides>,
    /// The overrides when the current drag began.
    gesture: Option<Overrides>,
    /// What Wobble's prefs held at the last load / save.
    saved: Overrides,
    status: Option<(String, f64)>,
    args_applied: bool,
}

impl App {
    fn new(ctx: &egui::Context) -> Self {
        ctx.set_theme(egui::Theme::Dark);
        ctx.global_style_mut(|s| {
            s.spacing.item_spacing = egui::vec2(8.0, 6.0);
            s.spacing.button_padding = egui::vec2(10.0, 4.0);
        });
        let prefs = read_prefs();
        theme::install(&theme_of(&prefs).unwrap_or_default());
        if let Some(sw) = swatches_of(&prefs) {
            colorpicker::set_swatches(sw);
        }
        Self {
            scene: Scene::Workspace,
            sel: Some(CORE[0].0),
            panel_hover: None,
            reveal: 0,
            view: colorpicker::View { mode: colorpicker::Mode::Wheel, scale: Vec2::splat(2.0), tabs_last: false },
            undo: Vec::new(),
            redo: Vec::new(),
            gesture: None,
            saved: theme::overrides(),
            status: None,
            args_applied: false,
        }
    }

    /// Dev flags for checking the app without driving it:
    /// `--scene N`, `--select slot_id`, `--set slot_id=#hex`, `--shot out.png`.
    fn apply_args(&mut self) {
        let args: Vec<String> = std::env::args().collect();
        let mut it = args.iter().skip(1);
        while let Some(a) = it.next() {
            let Some(v) = it.next() else { break };
            match a.as_str() {
                "--scene" => {
                    if let Some(s) = v.parse::<usize>().ok().and_then(|i| Scene::ALL.get(i)) {
                        self.scene = *s;
                    }
                }
                "--select" => {
                    // A node's own slot selects that node alone.
                    self.sel = DEFS
                        .iter()
                        .position(|d| d.id == v)
                        .map(|i| theme::ALL[i])
                        .and_then(|s| if node_of(s).is_some() { Some(s) } else { pick_of(s) });
                    self.reveal = 3;
                }
                "--set" => {
                    if let Some((id, hex)) = v.split_once('=') {
                        let mut m = theme::overrides();
                        m.insert(id.to_string(), hex.to_string());
                        theme::install(&m);
                    }
                }
                _ => {}
            }
        }
    }

    fn say(&mut self, ctx: &egui::Context, s: impl Into<String>) {
        self.status = Some((s.into(), ctx.input(|i| i.time)));
    }

    /// Before a one-shot edit (reset, import): remember where we were.
    fn checkpoint(&mut self) {
        self.undo.push(theme::overrides());
        self.redo.clear();
    }

    fn undo(&mut self) {
        if let Some(prev) = self.undo.pop() {
            self.redo.push(theme::overrides());
            theme::install(&prev);
        }
    }
    fn redo(&mut self) {
        if let Some(next) = self.redo.pop() {
            self.undo.push(theme::overrides());
            theme::install(&next);
        }
    }

    fn save_to_wobble(&mut self, ctx: &egui::Context) {
        let Some(path) = prefs_path() else {
            self.say(ctx, "no home folder: can't find ~/.wobble");
            return;
        };
        let mut prefs = read_prefs();
        if !prefs.is_object() {
            prefs = serde_json::json!({});
        }
        let over = theme::overrides();
        prefs["theme"] = serde_json::to_value(&over).unwrap();
        prefs["picker_swatches"] = serde_json::to_value(colorpicker::swatches()).unwrap();
        let res = path
            .parent()
            .map(std::fs::create_dir_all)
            .transpose()
            .and_then(|_| std::fs::write(&path, serde_json::to_string_pretty(&prefs).unwrap()));
        match res {
            Ok(()) => {
                self.saved = over;
                if wobble_running() {
                    self.say(ctx, "saved. Wobble is open: quit and reopen it to see the colours");
                } else {
                    self.say(ctx, "saved. Open Wobble to see it");
                }
            }
            Err(e) => self.say(ctx, format!("couldn't save: {e}")),
        }
    }

    fn export(&mut self, ctx: &egui::Context) {
        let Some(p) = rfd::FileDialog::new()
            .add_filter("wobble theme", &["json"])
            .set_file_name("wobble-theme.json")
            .save_file()
        else {
            return;
        };
        let doc = serde_json::json!({ "theme": theme::overrides() });
        match std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap()) {
            Ok(()) => self.say(ctx, format!("exported {}", p.display())),
            Err(e) => self.say(ctx, format!("couldn't export: {e}")),
        }
    }

    fn import(&mut self, ctx: &egui::Context) {
        let Some(p) = rfd::FileDialog::new().add_filter("wobble theme", &["json"]).pick_file() else {
            return;
        };
        let parsed = std::fs::read_to_string(&p)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| theme_of(&v));
        match parsed {
            Some(t) => {
                self.checkpoint();
                theme::install(&t);
                self.say(ctx, format!("imported {}", p.display()));
            }
            None => self.say(ctx, "that file has no theme in it"),
        }
    }

    // --- top bar ----------------------------------------------------------

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            ui.label(RichText::new("wobble colours").strong().size(15.0));
            ui.separator();
            for s in Scene::ALL {
                if ui.selectable_label(self.scene == s, s.label()).clicked() {
                    self.scene = s;
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let dirty = theme::overrides() != self.saved;
                let save = egui::Button::new(RichText::new("save to Wobble").strong())
                    .fill(if dirty { Color32::from_rgb(40, 96, 60) } else { ui.visuals().widgets.inactive.bg_fill });
                if ui.add(save).clicked() {
                    self.save_to_wobble(&ctx);
                }
                ui.menu_button("more", |ui| {
                    if ui.button("export to a file…").clicked() {
                        self.export(&ctx);
                    }
                    if ui.button("import a file…").clicked() {
                        self.import(&ctx);
                    }
                    ui.separator();
                    if ui.add_enabled(theme::any_overridden(), egui::Button::new("start over (Wobble's defaults)")).clicked() {
                        self.checkpoint();
                        theme::reset_all();
                    }
                });
            });
        });
    }

    // --- the colour list --------------------------------------------------

    fn colours(&mut self, ui: &mut egui::Ui) {
        self.panel_hover = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            ui.add_space(6.0);
            ui.label(RichText::new("pick these colours").size(16.0).strong());
            ui.weak("everything else in Wobble is a shade of one of them");
            ui.add_space(8.0);
            for &(slot, name, hint) in CORE {
                let open = self.sel == Some(slot);
                if self.row(ui, slot, name, hint, open).clicked() {
                    self.sel = if open { None } else { Some(slot) };
                }
                if open {
                    ui.indent(("picker", slot as usize), |ui| self.picker(ui, slot, "⟲ back to Wobble's default"));
                    ui.add_space(6.0);
                }
                ui.add_space(4.0);
            }

            ui.add_space(14.0);
            ui.label(RichText::new("node colours").size(16.0).strong());
            ui.weak("every node is a shade of its family. Click a node for its family, double-click it to tune just that node");
            ui.add_space(8.0);
            for &(slot, name, hint) in NODE_LINES {
                let open = self.sel == Some(slot);
                if self.row(ui, slot, name, hint, open).clicked() {
                    self.sel = if open { None } else { Some(slot) };
                }
                if open {
                    ui.indent(("picker", slot as usize), |ui| self.picker(ui, slot, "⟲ back to Wobble's default"));
                    ui.add_space(6.0);
                }
                ui.add_space(4.0);
            }
            for &(fam, name) in FAMILIES {
                let members: Vec<&glyphs::Node> =
                    glyphs::NODES.iter().filter(|n| family_of(n.slot) == Some(fam)).collect();
                let member = self.sel.filter(|&s| s != fam && members.iter().any(|n| n.slot == s));
                let open = self.sel == Some(fam) || member.is_some();
                let hint = names_hint(&members);
                if self.row(ui, fam, name, &hint, open).clicked() {
                    self.sel = if self.sel == Some(fam) { None } else { Some(fam) };
                }
                if open {
                    ui.indent(("family", fam as usize), |ui| {
                        if self.sel == Some(fam) {
                            self.picker(ui, fam, "⟲ back to Wobble's default");
                        }
                        self.members(ui, &members);
                        if let Some(m) = member {
                            ui.add_space(6.0);
                            ui.label(RichText::new(format!("{} only", name_of(m))).size(14.0).strong());
                            self.picker(ui, m, &format!("⟲ follow {name} again"));
                        }
                    });
                    ui.add_space(6.0);
                }
                ui.add_space(4.0);
            }
            // A node that follows no family (a double-click on ToData) gets a
            // row of its own while it is being tuned.
            if let Some(s) = self.sel.filter(|&s| node_of(s).is_some() && family_of(s).is_none()) {
                let from = DEFS[s as usize].base.parent().map_or("its own colour", |p| DEFS[p as usize].label);
                if self.row(ui, s, name_of(s), &format!("borrows {from}"), true).clicked() {
                    self.sel = None;
                }
                ui.indent(("borrowed", s as usize), |ui| self.picker(ui, s, "⟲ back to Wobble's default"));
            }
        });
        self.reveal = self.reveal.saturating_sub(1);
    }

    /// One pickable colour in the list: a swatch, a name, a hint.
    fn row(&mut self, ui: &mut egui::Ui, slot: Slot, name: &str, hint: &str, open: bool) -> egui::Response {
        let resp = ui
            .horizontal(|ui| {
                let (rect, _) = ui.allocate_exact_size(egui::vec2(36.0, 36.0), Sense::hover());
                ui.painter().rect_filled(rect, 6.0, theme::get(slot));
                let edge = if open {
                    egui::Stroke::new(2.0, Color32::from_rgb(255, 225, 60))
                } else {
                    egui::Stroke::new(1.0, Color32::from_gray(80))
                };
                ui.painter().rect_stroke(rect, 6.0, edge, egui::StrokeKind::Outside);
                ui.vertical(|ui| {
                    ui.label(RichText::new(name).size(14.0).strong());
                    ui.weak(hint);
                });
            })
            .response
            .interact(Sense::click());
        if resp.hovered() {
            self.panel_hover = Some(slot);
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if open && self.reveal > 0 {
            resp.scroll_to_me(Some(egui::Align::Min));
        }
        resp
    }

    /// A family's nodes as chips: click one to tune that node alone, again
    /// to go back to the family.
    fn members(&mut self, ui: &mut egui::Ui, members: &[&glyphs::Node]) {
        ui.add_space(4.0);
        ui.weak("tune one node:");
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
            for n in members {
                let on = self.sel == Some(n.slot);
                let ink = if on { Color32::from_rgb(255, 225, 60) } else { ui.visuals().text_color() };
                let galley = ui.painter().layout_no_wrap(n.name.to_string(), egui::FontId::proportional(12.5), ink);
                let (rect, resp) = ui.allocate_exact_size(egui::vec2(galley.size().x + 28.0, 22.0), Sense::click());
                let v = ui.visuals();
                let bg = if resp.hovered() { v.widgets.hovered.bg_fill } else { v.widgets.inactive.bg_fill };
                ui.painter().rect_filled(rect, 11.0, bg);
                if on {
                    ui.painter().rect_stroke(rect, 11.0, egui::Stroke::new(1.5, ink), egui::StrokeKind::Inside);
                }
                // An overridden node wears a ring: it no longer follows.
                let dot = rect.left_center() + egui::vec2(12.0, 0.0);
                ui.painter().circle_filled(dot, 6.0, theme::get(n.slot));
                let rim = if theme::is_overridden(n.slot) { ink } else { Color32::from_gray(20) };
                ui.painter().circle_stroke(dot, 6.0, egui::Stroke::new(1.0, rim));
                ui.painter().galley(rect.left_center() + egui::vec2(22.0, -galley.size().y * 0.5), galley, ink);
                if resp.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                if resp.clicked() {
                    self.sel = if on { family_of(n.slot) } else { Some(n.slot) };
                }
            }
        });
    }

    fn picker(&mut self, ui: &mut egui::Ui, slot: Slot, reset: &str) {
        let cur = theme::get(slot);
        let [r, g, b, al] = cur.to_srgba_unmultiplied();
        let byte = |x: u8| dec1(x as f32 / 255.0);
        let mut v = [byte(r), byte(g), byte(b), al as f32 / 255.0];
        let mut pal = colorpicker::swatches();
        let before = pal.clone();
        let mut geom = colorpicker::Geom::default();
        // Fit the picker to the panel's width: its content is
        // 60 + 172·x points wide (label + lane + number, colorpicker::Geo).
        let w = ui.available_width() - 4.0;
        self.view.scale = Vec2::new(((w - 60.0) / 172.0).clamp(colorpicker::SCALE_MIN, colorpicker::SCALE_MAX), 1.6);
        let id = egui::Id::new(("design-picker", slot as usize));
        if colorpicker::picker_ui(ui, id, &mut v, PickSpec::rgb(), &mut self.view, &mut pal, &mut geom) {
            let e = |x: f32| (enc1(x).clamp(0.0, 1.0) * 255.0).round() as u8;
            let c = Color32::from_rgba_unmultiplied(e(v[0]), e(v[1]), e(v[2]), al);
            if c != cur {
                if self.gesture.is_none() {
                    self.gesture = Some(theme::overrides());
                }
                theme::set(slot, c);
            }
        }
        if pal != before {
            colorpicker::set_swatches(pal);
        }
        if theme::is_overridden(slot) && ui.small_button(reset).clicked() {
            self.checkpoint();
            theme::reset(slot);
        }
    }

    // --- the layout -----------------------------------------------------------

    fn layout(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click());
        let time = ui.input(|i| i.time);
        let mut c = canvas::Canvas::new(ui.painter_at(rect), rect.shrink(8.0), time);
        self.scene.paint(&mut c);

        let hit = resp.hover_pos().and_then(|p| c.pick(p)).map(|i| c.hits[i].slot);
        let hit_pick = hit.and_then(pick_of);
        let hover = hit_pick.or(self.panel_hover);
        let lit = |t: Option<Slot>| -> Vec<bool> {
            theme::ALL.iter().map(|&s| t.is_some_and(|t| follows(s, t))).collect()
        };
        c.halos(&lit(self.sel), &lit(hover));
        if self.sel.is_some() {
            // The pulse.
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(33));
        }

        match hit_pick {
            Some(s) => {
                // A node: a click takes its family, a double-click the node.
                let node = hit.and_then(node_of).filter(|n| n.slot != s);
                let resp = resp.clone().on_hover_ui_at_pointer(|ui| match node {
                    Some(n) => {
                        ui.label(RichText::new(n.name).strong());
                        ui.weak(format!("click: {} · double-click: just {}", name_of(s), n.name));
                    }
                    None => {
                        ui.label(RichText::new(name_of(s)).strong());
                    }
                });
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                if let (true, Some(n)) = (resp.double_clicked(), node) {
                    self.sel = Some(n.slot);
                    self.reveal = 3;
                } else if resp.clicked() {
                    if self.sel != Some(s) {
                        self.reveal = 3;
                    }
                    self.sel = Some(s);
                }
            }
            None if resp.clicked() => self.sel = None,
            None => {}
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.frame(ui);
    }
}

impl App {
    /// One frame of the whole window: eframe's, or `--shot`'s with none.
    fn frame(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        if !self.args_applied {
            self.args_applied = true;
            self.apply_args();
        }
        let (redo, undo) = ctx.input_mut(|i| {
            let redo = i.consume_key(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z);
            (redo, i.consume_key(egui::Modifiers::COMMAND, egui::Key::Z))
        });
        if !colorpicker::any_open(&ctx) {
            if redo {
                self.redo();
            } else if undo {
                self.undo();
            }
        }

        egui::Panel::top("top").show(ui, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(2.0);
        });
        let now = ctx.input(|i| i.time);
        if let Some((s, t)) = self.status.clone().filter(|(_, t)| now - t < 8.0) {
            let _ = t;
            egui::Panel::bottom("status").show(ui, |ui| {
                ui.label(s);
            });
        }
        egui::Panel::right("colours")
            .default_size(380.0)
            .size_range(320.0..=520.0)
            .show(ui, |ui| self.colours(ui));
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(Color32::from_gray(14)))
            .show(ui, |ui| self.layout(ui));

        // A drag through the picker is one undo step.
        if self.gesture.is_some() && !ctx.input(|i| i.pointer.any_down()) {
            let before = self.gesture.take().unwrap();
            if before != theme::overrides() {
                self.undo.push(before);
                self.redo.clear();
            }
        }
    }
}
