//! The `colour…` window: per-format colour defaults and the default viewer
//! transform, saved for this machine (`~/.wobble/prefs.json`).
//!
//! - **input** — what a Read of each format holds. A Read left on "default"
//!   follows its row; picking a space on the Read overrides it.
//! - **output** — what a Write of each format encodes into, same rule.
//! - **viewer** — the view transform the viewer opens on; the viewer bar's
//!   `view` menu overrides it for this session.
//!
//! The working space is not here: it is ACEScg for every new comp and lives
//! with the setup (globals…).
//!
//! A child module of `main`, so it reaches `Gfx`'s private state directly.

use crate::{colorui, save_prefs, Gfx};
use engine::color::{Colorspace, FormatDefaults, InputSpace, INPUT_FORMATS, OUTPUT_FORMATS};
use engine::viewtf::{Eotf, Gamut, ViewPreset, ViewTransform};

fn heading(ui: &mut egui::Ui, s: &str) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(s).weak());
}

/// Which files a table row covers, for its hover.
fn format_hover(f: &str) -> &'static str {
    match f {
        "jpg" => ".jpg / .jpeg",
        "movie" => ".mov .mp4 .m4v .mxf .avi .mkv",
        "exr" => ".exr",
        "dpx" => ".dpx",
        "png" => ".png",
        _ => "",
    }
}

/// One input row: working (no transform) or an explicit space.
fn input_combo(ui: &mut egui::Ui, f: &str, v: &mut InputSpace, working: Colorspace) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(("colour-in", f))
        .width(220.0)
        .selected_text(v.label(working))
        .show_ui(ui, |ui| {
            changed |= ui
                .selectable_value(v, InputSpace::Working, format!("working ({})", working.label()))
                .on_hover_text("no transform — the files are already in the working space")
                .changed();
            ui.separator();
            for (name, cs) in Colorspace::PRESETS {
                changed |= ui.selectable_value(v, InputSpace::Explicit(cs), name).changed();
            }
        });
    changed
}

/// One output row: `None` = write the working space unconverted.
fn output_combo(
    ui: &mut egui::Ui,
    f: &str,
    v: &mut Option<Colorspace>,
    working: Colorspace,
) -> bool {
    let mut changed = false;
    let label = match v {
        None => format!("working ({})", working.label()),
        Some(cs) => cs.label(),
    };
    egui::ComboBox::from_id_salt(("colour-out", f))
        .width(220.0)
        .selected_text(label)
        .show_ui(ui, |ui| {
            changed |= ui
                .selectable_value(v, None, format!("working ({})", working.label()))
                .on_hover_text("no transform — write the comp's own scene-linear values")
                .changed();
            ui.separator();
            for (name, cs) in Colorspace::PRESETS {
                changed |= ui.selectable_value(v, Some(cs), name).changed();
            }
        });
    changed
}

/// Why a menu entry cannot show what it says, if it cannot.
fn unavailable_hint(t: ViewTransform) -> Option<&'static str> {
    match t {
        _ if t.available() => None,
        ViewTransform::AgX => Some(
            "no Blender install found — AgX reads Blender's own LUTs \
             (set WOBBLE_BLENDER_LUTS to its datafiles/colormanagement/luts)",
        ),
        _ => Some("rebuild with --features opendrt (GPLv3)"),
    }
}

/// One menu entry, greyed with a reason when this build/machine cannot
/// show it. Shared with the viewer bar.
pub(crate) fn preset_entry(
    ui: &mut egui::Ui,
    cur: &mut Option<ViewPreset>,
    p: ViewPreset,
) -> bool {
    let mut r = ui.selectable_value(cur, Some(p), p.label());
    if let Some(why) = unavailable_hint(p.transform) {
        r = r.on_hover_text(why);
    }
    r.changed()
}

/// What the panel changed this frame.
#[derive(Default)]
pub(crate) struct Changed {
    pub defaults: bool,
    /// The default viewer preset.
    pub view: bool,
    /// Which presets the viewer bar offers.
    pub menu: bool,
}

/// The viewer section: which presets the viewer bar's menu offers (the
/// catalogue plus any custom ones), and which is the default.
fn viewer_section(
    ui: &mut egui::Ui,
    menu: &mut Vec<ViewPreset>,
    dflt: &mut ViewPreset,
    viewer_override: Option<ViewPreset>,
    ch: &mut Changed,
) {
    heading(ui, "viewer menu — tick what the viewer bar offers; the dot marks the default");
    let mut rows: Vec<ViewPreset> = ViewPreset::CATALOGUE.iter().map(|(_, p)| *p).collect();
    let custom: Vec<ViewPreset> = menu.iter().copied().filter(|p| !rows.contains(p)).collect();
    rows.extend(custom);
    let mut remove = None;
    egui::Grid::new("colour-view-menu").num_columns(3).spacing([8.0, 2.0]).show(ui, |ui| {
        for p in rows {
            let custom = !ViewPreset::CATALOGUE.iter().any(|(_, c)| *c == p);
            let mut on = menu.contains(&p);
            let mut label = egui::RichText::new(p.label());
            if !p.transform.available() {
                label = label.weak();
            }
            let mut r = ui.checkbox(&mut on, label);
            if let Some(why) = unavailable_hint(p.transform) {
                r = r.on_hover_text(why);
            }
            if r.changed() {
                if on {
                    menu.push(p);
                } else if menu.len() > 1 {
                    menu.retain(|m| *m != p);
                    if *dflt == p {
                        *dflt = menu[0];
                        ch.view = true;
                    }
                }
                ch.menu = true;
            }
            if ui
                .add_enabled(menu.contains(&p), egui::RadioButton::new(*dflt == p, ""))
                .on_hover_text("the viewer opens on this")
                .clicked()
                && *dflt != p
            {
                *dflt = p;
                ch.view = true;
            }
            if custom {
                if ui.small_button("×").on_hover_text("remove this custom entry").clicked() {
                    remove = Some(p);
                }
            } else {
                ui.label("");
            }
            ui.end_row();
        }
    });
    if let Some(p) = remove {
        if menu.len() > 1 {
            menu.retain(|m| *m != p);
            if *dflt == p {
                *dflt = menu[0];
                ch.view = true;
            }
            ch.menu = true;
        }
    }

    // A custom entry: any transform on any display + curve.
    let id = ui.make_persistent_id("colour-view-custom");
    let mut c: ViewPreset = ui.data(|d| d.get_temp(id)).unwrap_or_default();
    ui.horizontal(|ui| {
        ui.weak("custom");
        egui::ComboBox::from_id_salt("cv-tf").width(90.0).selected_text(c.transform.short()).show_ui(
            ui,
            |ui| {
                for t in ViewTransform::ALL {
                    ui.selectable_value(&mut c.transform, t, t.short());
                }
            },
        );
        egui::ComboBox::from_id_salt("cv-gamut").width(110.0).selected_text(c.display.label()).show_ui(
            ui,
            |ui| {
                for g in Gamut::DISPLAYS {
                    ui.selectable_value(&mut c.display, g, g.label());
                }
            },
        );
        egui::ComboBox::from_id_salt("cv-eotf").width(100.0).selected_text(c.eotf.label()).show_ui(
            ui,
            |ui| {
                for e in Eotf::ALL {
                    ui.selectable_value(&mut c.eotf, e, e.label());
                }
            },
        );
        if ui.add_enabled(!menu.contains(&c), egui::Button::new("add")).clicked() {
            menu.push(c);
            ch.menu = true;
        }
    });
    ui.data_mut(|d| d.insert_temp(id, c));
    if let Some(o) = viewer_override {
        ui.weak(format!("the viewer is overriding: {}", o.label()));
    }
}

/// The window's body, free of `Gfx` so a headless shot can draw it.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    d: &mut FormatDefaults,
    menu: &mut Vec<ViewPreset>,
    dflt: &mut ViewPreset,
    viewer_override: Option<ViewPreset>,
    working: Colorspace,
) -> Changed {
    let mut ch = Changed::default();
    ui.weak("defaults for this machine — every setup; a node's own pick wins");

    heading(ui, "input — what each format's files hold");
    egui::Grid::new("colour-in-grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        for f in INPUT_FORMATS {
            ui.label(f).on_hover_text(format_hover(f));
            let mut v = d.input_for_format(f);
            if input_combo(ui, f, &mut v, working) {
                d.input.insert(f.to_string(), v);
                ch.defaults = true;
            }
            ui.end_row();
        }
    });

    heading(ui, "output — what Renders encode into");
    egui::Grid::new("colour-out-grid").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
        for f in OUTPUT_FORMATS {
            ui.label(f).on_hover_text(format_hover(f));
            let mut v = d.output_for_format(f);
            if output_combo(ui, f, &mut v, working) {
                d.output.insert(f.to_string(), v);
                ch.defaults = true;
            }
            ui.end_row();
        }
    });

    viewer_section(ui, menu, dflt, viewer_override, &mut ch);

    ui.separator();
    ui.horizontal(|ui| {
        ui.weak(format!("working: {} (per setup, globals…)", working.label()));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add_enabled(*d != FormatDefaults::default(), egui::Button::new("reset formats"))
                .on_hover_text("back to the built-in format defaults")
                .clicked()
            {
                *d = FormatDefaults::default();
                ch.defaults = true;
            }
        });
    });
    ch
}

impl Gfx {
    /// The presets the viewer bar offers (prefs, else the default menu).
    pub(crate) fn view_menu(&self) -> Vec<ViewPreset> {
        self.prefs.view_menu.clone().unwrap_or_else(ViewPreset::default_menu)
    }

    pub(crate) fn colour_window(&mut self, ctx: &egui::Context) {
        if !self.colour_ui_open {
            return;
        }
        let working = self.egraph.working;
        let mut d = colorui::defaults();
        let mut menu = self.view_menu();
        let mut dflt = self.view.preset();
        let over = self.viewer_tf;
        let mut ch = Changed::default();
        let mut open = true;
        egui::Window::new("colour")
            .open(&mut open)
            .default_width(420.0)
            .vscroll(true)
            .show(ctx, |ui| {
                ch = panel(ui, &mut d, &mut menu, &mut dflt, over, working);
            });
        self.colour_ui_open = open;
        if ch.view {
            self.view = self.view.with_preset(dflt);
            self.prefs.view = Some(self.view);
        }
        if ch.menu {
            self.prefs.view_menu = Some(menu);
        }
        if ch.view || ch.menu {
            save_prefs(&self.prefs);
        }
        if ch.defaults {
            self.set_colour_defaults(d);
        }
    }

    /// Install a new defaults table: every Read / Write on "default" now
    /// resolves differently, which recompiles their input transforms.
    pub(crate) fn set_colour_defaults(&mut self, d: FormatDefaults) {
        colorui::set_defaults(d.clone());
        self.prefs.color_defaults = Some(d);
        save_prefs(&self.prefs);
        self.playback_dirty = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Look at it: `cargo test -p app colour_window_shot -- --ignored`,
    /// then `target/colour-window.png`. A Rec.709 dpx row, the viewer
    /// overridden, and a Read + Write row under it on "default".
    #[test]
    #[ignore]
    fn colour_window_shot() {
        let mut d = FormatDefaults::default();
        d.input.insert("dpx".into(), InputSpace::Explicit(Colorspace::REC709));
        let mut menu = ViewPreset::default_menu();
        let mut dflt = menu[0];
        let over = Some(ViewPreset::CATALOGUE[2].1);
        crate::uishot::render("colour-window", 480.0, 700.0, 2.0, 3, |ui, _| {
            panel(ui, &mut d, &mut menu, &mut dflt, over, Colorspace::ACESCG);
            ui.separator();
            let mut input = InputSpace::Default;
            let mut out = None;
            let mut follow = true;
            colorui::input_space_row(ui, "r", &mut input, Colorspace::ACESCG, "p/sh010.####.png");
            colorui::output_space_row(ui, "w", &mut out, &mut follow, Colorspace::ACESCG, "o/c.####.exr");
        })
        .expect("headless shot");
    }
}
