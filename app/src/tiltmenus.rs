//! The top bar's three settings menus as [`tiltmenu`] chips: prefs, globals
//! and colour. Built from `Gfx` state every frame, applied from the events
//! that come back — the widget itself knows nothing about wobble.
//!
//! These carry the settings a hand reaches for; each menu ends in an "all …"
//! row that opens the full window (`settingsui`, `colorprefs`), which stays
//! the complete record.
//!
//! A child module of `main`, so it reaches `Gfx`'s private state directly.

use crate::tiltmenu::{Event, Item, Menu};
use crate::{colorui, editor, save_prefs, tools, Gfx};
use engine::color::{Colorspace, InputSpace, INPUT_FORMATS};

/// Scene-linear presets: what a working space may be.
fn working_presets() -> Vec<(&'static str, Colorspace)> {
    Colorspace::PRESETS.into_iter().filter(|(_, cs)| cs.is_scene_linear()).collect()
}

/// "working", then every preset: an input or output default's options.
fn space_options(working: Colorspace) -> Vec<String> {
    std::iter::once(format!("working ({})", working.label()))
        .chain(Colorspace::PRESETS.iter().map(|(n, _)| n.to_string()))
        .collect()
}

const OUT_EXR: &str = "out-exr";

impl Gfx {
    pub(crate) fn tilt_menus(&self) -> Vec<Menu> {
        let fetching = self.model_downloads.busy();
        let prefs = Menu {
            id: "prefs",
            short: if fetching { "prefs ⏺".into() } else { "prefs".into() },
            full: if fetching { "preferences ⏺".into() } else { "preferences".into() },
            items: vec![
                Item::sub(
                    "graph",
                    vec![
                        Item::check(
                            "links",
                            "value links over the graph",
                            self.prefs.show_patch_links.unwrap_or(false),
                        ),
                        Item::check("rgb", "collapse rgb triples", !self.prefs.expand_rgb.unwrap_or(false)),
                        Item::slider(
                            "magnet",
                            "magnet reach",
                            self.prefs.magnet_reach.unwrap_or(editor::MAGNET_REACH),
                            0.0,
                            200.0,
                        )
                        .step(1.0)
                        .unit("u"),
                        Item::slider("selcap", "selection cap", self.sel_cap as f32, 0.0, 64.0).step(1.0),
                    ],
                ),
                Item::sub(
                    "viewer",
                    vec![
                        Item::slider("gain", "gain", self.view.exposure, -6.0, 6.0).unit(" st"),
                        Item::slider("gamma", "gamma", self.view.gamma, 0.2, 4.0),
                    ],
                ),
                Item::action("colours", "interface colours…"),
                Item::action("keys", "keys…"),
                Item::action("models", if fetching { "models… ⏺" } else { "models…" }),
                Item::action("reload", "reload plugins + subwobs"),
                Item::action("all-prefs", "all preferences…"),
            ],
        };

        let (a, b) = self.script_range();
        let working = self.egraph.working;
        let wp = working_presets();
        let statuses: Vec<_> = editor::NodeStatus::ALL.iter().rev().copied().collect();
        let globals = Menu {
            id: "globals",
            short: "globals".into(),
            full: "global settings".into(),
            items: vec![
                Item::slider("fps", "fps", self.fps, 1.0, 120.0).step(1.0),
                Item::check("range", format!("set range ({a}–{b})"), self.comp_range.is_some()),
                Item::choice(
                    "working",
                    "working space",
                    wp.iter().map(|(n, _)| n.to_string()).collect(),
                    wp.iter().position(|(_, cs)| *cs == working),
                ),
                Item::choice(
                    "offer",
                    "add menus offer",
                    statuses.iter().map(|s| s.filter_label().to_string()).collect(),
                    statuses.iter().position(|s| *s == self.egraph.node_status_min),
                ),
                Item::action("all-globals", "all globals…"),
            ],
        };

        let d = colorui::defaults();
        let menu = self.view_menu();
        let dflt = self.view.preset();
        let opts = space_options(working);
        let pick_in = |v: &InputSpace| match v {
            InputSpace::Explicit(cs) => {
                Colorspace::PRESETS.iter().position(|(_, p)| p == cs).map(|i| i + 1)
            }
            _ => Some(0),
        };
        let inputs = INPUT_FORMATS
            .iter()
            .map(|f| {
                let sel = d.input.get(*f).map_or(Some(0), pick_in);
                Item::choice(f, *f, opts.clone(), sel)
            })
            .collect();
        let out_sel = match d.output.get("exr").copied().flatten() {
            None => Some(0),
            Some(cs) => Colorspace::PRESETS.iter().position(|(_, p)| *p == cs).map(|i| i + 1),
        };
        // Blender's looks, for the transform the viewer is showing.
        let looks = self.shown_view().transform.looks();
        let look = (looks.len() > 1).then(|| {
            Item::choice(
                "look",
                "look",
                looks.iter().map(|l| l.label().to_string()).collect(),
                Some(looks.iter().position(|l| *l == self.view.look).unwrap_or(0)),
            )
        });
        let colour = Menu {
            id: "colour",
            short: "colour".into(),
            full: "colour management".into(),
            items: vec![
                Some(Item::choice(
                    "viewer",
                    "viewer opens on",
                    menu.iter().map(|p| p.label()).collect(),
                    menu.iter().position(|p| *p == dflt),
                )),
                look,
                Some(Item::sub("input, per format", inputs)),
                Some(Item::sub("output", vec![Item::choice(OUT_EXR, "exr", opts, out_sel)])),
                Some(Item::action("all-colour", "all colour settings…")),
            ]
            .into_iter()
            .flatten()
            .collect(),
        };
        let mut addon_items = crate::addons::menu_items(&self.addons);
        addon_items.push(Item::action("addons", "addons…"));
        let addons = Menu {
            id: "addons",
            short: "addons".into(),
            full: "addons".into(),
            items: addon_items,
        };
        vec![prefs, globals, colour, addons]
    }

    pub(crate) fn apply_tilt(&mut self, events: Vec<Event>) {
        let mut prefs_dirty = false;
        for e in events {
            match e {
                Event::Toggle("links", v) => {
                    self.prefs.show_patch_links = Some(v);
                    prefs_dirty = true;
                }
                Event::Toggle("rgb", v) => {
                    self.prefs.expand_rgb = Some(!v);
                    prefs_dirty = true;
                }
                Event::Slide("magnet", v) => {
                    self.prefs.magnet_reach = Some(v);
                    prefs_dirty = true;
                }
                Event::Slide("selcap", v) => {
                    self.sel_cap = v as usize;
                    self.prefs.sel_cap = Some(self.sel_cap);
                    prefs_dirty = true;
                }
                Event::Slide("gain", v) => self.view.exposure = v,
                Event::Slide("gamma", v) => self.view.gamma = v,
                Event::Action("colours") => self.theme_ui.open = true,
                Event::Action("keys") => self.keys_ui.open = true,
                Event::Action("models") => self.models_ui_open = true,
                Event::Action("reload") => {
                    engine::plugins::clear();
                    tools::clear();
                    crate::load_plugins();
                    self.addons_reload();
                    // The reload cleared the comp's own subwobs too.
                    self.sync_subwobs();
                }
                Event::Action("addons") => self.addons.window_open = true,
                Event::Action(id) | Event::Toggle(id, _) if crate::addons::menu_hit(id).is_some() => {
                    match crate::addons::menu_hit(id) {
                        Some(crate::addons::MenuHit::Command(a, c)) => self.addons.command(&a, &c),
                        Some(crate::addons::MenuHit::Panel(a, p)) => {
                            if let Some(ad) = self.addons.get_mut(&a) {
                                if !ad.open_panels.remove(&p) {
                                    ad.open_panels.insert(p);
                                }
                            }
                        }
                        None => {}
                    }
                }
                Event::Action("all-prefs") => self.prefs_ui_open = true,

                Event::Slide("fps", v) => self.fps = v,
                Event::Toggle("range", on) => {
                    let (a, b) = self.script_range();
                    self.comp_range = on.then_some((a, b));
                }
                Event::Pick("working", i) => {
                    if let Some((_, cs)) = working_presets().get(i) {
                        self.egraph.working = *cs;
                        self.playback_dirty = true;
                    }
                }
                Event::Pick("offer", i) => {
                    if let Some(s) = editor::NodeStatus::ALL.iter().rev().nth(i) {
                        self.egraph.node_status_min = *s;
                    }
                }
                Event::Action("all-globals") => self.globals_ui_open = true,

                Event::Pick("look", i) => {
                    if let Some(l) = self.shown_view().transform.looks().get(i) {
                        self.view.look = *l;
                        self.prefs.view = Some(self.view);
                        prefs_dirty = true;
                    }
                }
                Event::Pick("viewer", i) => {
                    if let Some(p) = self.view_menu().get(i) {
                        self.view = self.view.with_preset(*p);
                        self.prefs.view = Some(self.view);
                        prefs_dirty = true;
                    }
                }
                Event::Pick(f, i) if INPUT_FORMATS.contains(&f) => {
                    let v = match i {
                        0 => InputSpace::Working,
                        i => match Colorspace::PRESETS.get(i - 1) {
                            Some((_, cs)) => InputSpace::Explicit(*cs),
                            None => continue,
                        },
                    };
                    let mut d = colorui::defaults();
                    d.input.insert(f.to_string(), v);
                    self.set_colour_defaults(d);
                }
                Event::Pick(OUT_EXR, i) => {
                    let v = match i {
                        0 => None,
                        i => match Colorspace::PRESETS.get(i - 1) {
                            Some((_, cs)) => Some(*cs),
                            None => continue,
                        },
                    };
                    let mut d = colorui::defaults();
                    d.output.insert("exr".into(), v);
                    self.set_colour_defaults(d);
                }
                Event::Action("all-colour") => self.colour_ui_open = true,
                _ => {}
            }
        }
        if prefs_dirty {
            save_prefs(&self.prefs);
        }
    }
}
