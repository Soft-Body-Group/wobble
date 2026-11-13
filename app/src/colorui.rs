//! Colour-management controls: what a plate is, what the comp works in.
//!
//! Deliberately small and shared. Every one of these pickers changes what
//! pixels MEAN rather than how they look, so each shows its current value in
//! full — "working (ACEScg)", not a blank — because the failure mode for all
//! of them is a wrong setting that still renders a plausible picture.

use engine::color::{Colorspace, FormatDefaults, InputSpace};
use std::collections::HashMap;
use std::sync::RwLock;

/// The per-format colour defaults in force (the `colour…` window edits
/// them, prefs persist them). A process global rather than a field because
/// every `EditorGraph` — the comp's, each edit-mode shot's, a bake's —
/// resolves `InputSpace::Default` through it when lowering to the engine,
/// and threading a table through all of those buys nothing.
static DEFAULTS: RwLock<Option<FormatDefaults>> = RwLock::new(None);

pub fn defaults() -> FormatDefaults {
    DEFAULTS.read().ok().and_then(|d| d.clone()).unwrap_or_default()
}

pub fn set_defaults(d: FormatDefaults) {
    if let Ok(mut g) = DEFAULTS.write() {
        *g = Some(d);
    }
}

/// Directory -> the format of the first decodable file in it, memoised: a
/// Read's path is usually a DIRECTORY of frames, and this runs every time a
/// graph is lowered.
static DIR_FORMATS: RwLock<Option<HashMap<String, Option<&'static str>>>> = RwLock::new(None);

/// The defaults-table row (`color::format_key`) a Read's path belongs to:
/// its own extension, else — a directory — the first file inside that has
/// one.
pub fn format_of(path: &str) -> Option<&'static str> {
    let p = std::path::Path::new(path);
    if let Some(f) = p.extension().and_then(|e| e.to_str()).and_then(engine::color::format_key) {
        return Some(f);
    }
    if path.is_empty() {
        return None;
    }
    if let Some(hit) = DIR_FORMATS.read().ok().and_then(|m| m.as_ref()?.get(path).copied()) {
        return hit;
    }
    let found = std::fs::read_dir(p).ok().and_then(|rd| {
        rd.flatten().find_map(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                return None;
            }
            std::path::Path::new(&*name)
                .extension()
                .and_then(|x| x.to_str())
                .and_then(engine::color::format_key)
        })
    });
    // Only a dir that exists is memoised: a path being typed, or footage
    // not yet on disk, is asked again next time.
    if p.is_dir() {
        if let Ok(mut m) = DIR_FORMATS.write() {
            m.get_or_insert_with(HashMap::new).insert(path.to_string(), found);
        }
    }
    found
}

/// What a Read of `path` assumes when set to `InputSpace::Default`.
pub fn default_input(path: &str) -> InputSpace {
    format_of(path).map_or(InputSpace::Working, |f| defaults().input_for_format(f))
}

/// `input` with `Default` resolved against the table — what the engine gets.
pub fn resolve_input(input: InputSpace, path: &str) -> InputSpace {
    match input {
        InputSpace::Default => default_input(path),
        v => v,
    }
}

/// The comp's working space. Only scene-linear spaces are offered: a comp
/// cannot work in a log encoding, and offering one would invite a setting
/// that quietly breaks every filter, blur and merge downstream.
pub fn working_row(ui: &mut egui::Ui, working: &mut Colorspace) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("working");
        egui::ComboBox::from_id_salt("comp-working")
            .selected_text(working.label())
            .show_ui(ui, |ui| {
                for (name, cs) in Colorspace::PRESETS {
                    if !cs.is_scene_linear() {
                        continue;
                    }
                    changed |= ui.selectable_value(working, cs, name).changed();
                }
            });
    })
    .response
    .on_hover_text(
        "the space every node works in, between the Reads' input transforms \
         and the Writes' output transforms",
    );
    changed
}

/// What a source node's plate is. `default` — the per-format table's answer
/// for this `path` — is first; the label always shows what that resolves to,
/// because a wrong setting here still renders a plausible picture.
pub fn input_space_row(
    ui: &mut egui::Ui,
    id_salt: &str,
    input: &mut InputSpace,
    working: Colorspace,
    path: &str,
) -> bool {
    let mut changed = false;
    let dflt = default_input(path);
    let dflt_label = format!("default ({})", dflt.label(working));
    ui.horizontal(|ui| {
        ui.label("input");
        let current = match *input {
            InputSpace::Default => dflt_label.clone(),
            v => v.label(working),
        };
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(current)
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(input, InputSpace::Default, &dflt_label)
                    .on_hover_text("follow the per-format default in colour…")
                    .changed();
                changed |= ui
                    .selectable_value(
                        input,
                        InputSpace::Working,
                        format!("working ({})", working.label()),
                    )
                    .on_hover_text("no transform — the plate is already in the working space")
                    .changed();
                ui.separator();
                for (name, cs) in Colorspace::PRESETS {
                    changed |= ui
                        .selectable_value(input, InputSpace::Explicit(cs), name)
                        .changed();
                }
            });
    })
    .response
    .on_hover_text("what this plate's pixels ARE; converted into the working space on read");
    changed
}

/// A colourspace-or-working label (`None` = the working space).
fn space_label(cs: Option<Colorspace>, working: Colorspace) -> String {
    match cs {
        None => format!("working ({})", working.label()),
        Some(cs) => cs.label(),
    }
}

/// The space a Write encodes into. `follow` = the per-format default for
/// `path`; otherwise `output`, where `None` means "the working space" —
/// for an EXR the honest choice: scene-linear, unconverted.
pub fn output_space_row(
    ui: &mut egui::Ui,
    id_salt: &str,
    output: &mut Option<Colorspace>,
    follow: &mut bool,
    working: Colorspace,
    path: &str,
) -> bool {
    let mut changed = false;
    let dflt_label = format!("default ({})", space_label(defaults().output_for(path), working));
    ui.horizontal(|ui| {
        ui.label("output");
        let label = if *follow { dflt_label.clone() } else { space_label(*output, working) };
        // One list over (follow, output): `None` is "default", Some(x) an
        // explicit pick (where x = None is the working space).
        let mut pick: Option<Option<Colorspace>> = (!*follow).then_some(*output);
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(label)
            .show_ui(ui, |ui| {
                changed |= ui
                    .selectable_value(&mut pick, None, &dflt_label)
                    .on_hover_text("follow the per-format default in colour…")
                    .changed();
                changed |= ui
                    .selectable_value(
                        &mut pick,
                        Some(None),
                        format!("working ({})", working.label()),
                    )
                    .on_hover_text("no transform — write the comp's own scene-linear values")
                    .changed();
                ui.separator();
                for (name, cs) in Colorspace::PRESETS {
                    changed |= ui.selectable_value(&mut pick, Some(Some(cs)), name).changed();
                }
            });
        if changed {
            *follow = pick.is_none();
            if let Some(o) = pick {
                *output = o;
            }
        }
    })
    .response
    .on_hover_text("the space the written files are encoded in");
    changed
}
