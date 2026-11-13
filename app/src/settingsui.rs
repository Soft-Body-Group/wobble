//! The two settings windows the top bar opens, split by where the setting
//! is SAVED:
//!
//! - **prefs…** — this machine and this hand, every setup
//!   (`~/.wobble/prefs.json`): colours, keys, cache, models, the display
//!   transform, graph behaviour.
//! - **globals…** — this setup, saved with it: script format, frame range,
//!   fps, working space, which nodes the add menus offer.
//!
//! A child module of `main`, so it reaches `Gfx`'s private state directly.

use crate::{colorui, editor, pickerui, report, save_prefs, theme, tools, viewui, widgets, Gfx};

/// A section heading, the same weak caps every window here uses.
fn heading(ui: &mut egui::Ui, s: &str) {
    ui.add_space(4.0);
    ui.label(egui::RichText::new(s).weak());
}

impl Gfx {
    pub(crate) fn prefs_window(&mut self, ctx: &egui::Context) {
        #[cfg(not(target_arch = "wasm32"))]
        self.network_window(ctx);
        if !self.prefs_ui_open {
            return;
        }
        let mut open = true;
        egui::Window::new("preferences")
            .open(&mut open)
            .default_width(440.0)
            .vscroll(true)
            .show(ctx, |ui| {
                ui.weak("saved for this machine — every setup");

                heading(ui, "interface");
                ui.horizontal(|ui| {
                    if ui.button("colours…").on_hover_text("recolour the interface").clicked() {
                        self.theme_ui.open = !self.theme_ui.open;
                    }
                    if ui.button("keys…").on_hover_text("remap hotkeys").clicked() {
                        self.keys_ui.open = !self.keys_ui.open;
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    if ui.button("network…").on_hover_text("other wobbles on this network: share this one, pair with others").clicked() {
                        crate::apiserve::peers::toggle_window();
                    }
                });
                ui.separator();

                heading(ui, "graph");
                // The patch's value cables, drawn over the graph but not
                // editable there — the patch bay owns the wiring.
                let mut plinks = self.prefs.show_patch_links.unwrap_or(false);
                if ui
                    .checkbox(&mut plinks, "value links over the graph")
                    .on_hover_text(
                        "draw the expression patch's links over the node graph\n\
                         read-only: wire them in the patch bay",
                    )
                    .changed()
                {
                    self.prefs.show_patch_links = Some(plinks);
                    save_prefs(&self.prefs);
                }
                // rgb triples: one port per parameter, or the three lanes the
                // model actually holds. Governs every expression view.
                let mut collapse = !self.prefs.expand_rgb.unwrap_or(false);
                if ui
                    .checkbox(&mut collapse, "collapse rgb triples")
                    .on_hover_text(
                        "rgb channel triples as ONE port carrying a three-strand \
                         cable\noff reveals the real per-lane links (they were always there)",
                    )
                    .changed()
                {
                    self.prefs.expand_rgb = Some(!collapse);
                    save_prefs(&self.prefs);
                }
                ui.horizontal(|ui| {
                    // Shift-to-join reach. 0 turns socket magnetism off.
                    let mut reach = self.prefs.magnet_reach.unwrap_or(editor::MAGNET_REACH);
                    if ui
                        .add(
                            egui::DragValue::new(&mut reach).update_while_editing(false)
                                .speed(widgets::rated(ui, 1.0))
                                .range(0.0..=200.0)
                                .prefix("magnet ")
                                .suffix("u"),
                        )
                        .on_hover_text(
                            "how close two sockets come before they court\n\
                             shift wires the courting pair; 0 = magnetism off",
                        )
                        .changed()
                    {
                        self.prefs.magnet_reach = Some(reach);
                        save_prefs(&self.prefs);
                    }
                    if ui
                        .add(
                            egui::DragValue::new(&mut self.sel_cap).update_while_editing(false)
                                .speed(widgets::rated(ui, 0.1))
                                .range(0..=64)
                                .prefix("sel cap "),
                        )
                        .on_hover_text("max selected nodes (0 = unlimited)")
                        .changed()
                    {
                        self.prefs.sel_cap = Some(self.sel_cap);
                        save_prefs(&self.prefs);
                    }
                });
                ui.separator();

                heading(ui, "display");
                if viewui::view_settings_ui(ui, &mut self.view) {
                    self.prefs.view = Some(self.view);
                    save_prefs(&self.prefs);
                }
                pickerui::panel_ui(ui, &mut self.picker);
                ui.separator();

                heading(ui, "cache");
                self.cache_settings_ui(ui);
                ui.separator();

                heading(ui, "alpha testing + updates");
                if report::sharing_ui(ui, &mut self.feedback) {
                    self.sync_report_prefs();
                    report::start_uploads();
                }
                ui.separator();

                heading(ui, "scripting API");
                self.api_settings_ui(ui);
                ui.separator();

                heading(ui, "models + plugins");
                ui.horizontal(|ui| {
                    let fetching = self.model_downloads.busy();
                    if ui
                        .button(if fetching { "models… ⏺" } else { "models…" })
                        .on_hover_text("ML weights: what is cached, and download the rest")
                        .clicked()
                    {
                        self.models_ui_open = !self.models_ui_open;
                    }
                    let (nplug, ntool) = (engine::plugins::all().len(), tools::all().len());
                    if ui
                        .button(format!("reload plugins ({nplug}) + subwobs ({ntool})"))
                        .on_hover_text(
                            "plugins/ + ~/.wobble/plugins\n\
                             a changed shader or subwob body recompiles its consumers",
                        )
                        .clicked()
                    {
                        // Authoritative: a deleted manifest stops offering
                        // its node rather than lingering until restart.
                        engine::plugins::clear();
                        tools::clear();
                        crate::load_plugins();
                        // The comp's own subwobs were cleared with the rest.
                        self.sync_subwobs();
                    }
                });
            });
        self.prefs_ui_open = open;
    }

    pub(crate) fn globals_window(&mut self, ctx: &egui::Context) {
        if !self.globals_ui_open {
            return;
        }
        let mut open = true;
        let name = self
            .current_setup
            .as_ref()
            .and_then(|p| p.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "untitled".into());
        egui::Window::new("globals")
            .open(&mut open)
            .default_width(380.0)
            .show(ctx, |ui| {
                ui.weak(format!("saved with {name}"));

                heading(ui, "format");
                // The script's format: what a generator set to "inherit"
                // (and anything with no picture input) renders at. Nodes
                // fed a plate keep the plate's size.
                let plates = self
                    .loaded
                    .values()
                    .map(|l| (l.seq.info.width, l.seq.info.height))
                    .reduce(|a, b| (a.0.max(b.0), a.1.max(b.1)))
                    .unwrap_or((1920, 1080));
                let before = self.canvas_override;
                widgets::format_row(ui, "script-format", &mut self.canvas_override, plates);
                if self.canvas_override != before {
                    self.playback_dirty = true;
                }
                ui.weak("generators set to inherit render at this; unset, the plates' size");
                ui.separator();

                heading(ui, "frames");
                // The SCRIPT's frame range: generators run for it unless they
                // set their own, and a shot delivers it. The viewer range is
                // the transport's — it follows whatever you are viewing.
                let (a, b) = self.script_range();
                let mut set = self.comp_range.is_some();
                if ui
                    .checkbox(&mut set, "set range")
                    .on_hover_text("unset, the script spans the loaded plates")
                    .changed()
                {
                    self.comp_range = set.then_some((a, b));
                }
                match self.comp_range {
                    Some((mut ra, mut rb)) => {
                        let mut ch = false;
                        ui.horizontal(|ui| {
                            ch |= ui.add(egui::DragValue::new(&mut ra).update_while_editing(false).prefix("first ")).changed();
                            ch |= ui.add(egui::DragValue::new(&mut rb).update_while_editing(false).prefix("last ")).changed();
                        });
                        if ch {
                            self.comp_range = Some((ra, rb.max(ra)));
                        }
                    }
                    None => {
                        let from = if self.media_union().is_some() { "the plates" } else { "default" };
                        ui.weak(format!("{a}–{b} ({from})"));
                    }
                }
                ui.add(
                    egui::DragValue::new(&mut self.fps).update_while_editing(false)
                        .speed(widgets::rated(ui, 0.2))
                        .range(1.0..=120.0)
                        .suffix(" fps"),
                );
                ui.separator();

                heading(ui, "colour");
                // Changing it recompiles every input transform: a topology
                // change, and it moves the render-cache key.
                if colorui::working_row(ui, &mut self.egraph.working) {
                    self.playback_dirty = true;
                }
                ui.separator();

                heading(ui, "nodes");
                // Which nodes the add menus offer, by maturity. Rides in the
                // setup so the choice travels with the comp.
                let mut min = self.egraph.node_status_min;
                egui::ComboBox::from_id_salt("node-status-min")
                    .selected_text(
                        egui::RichText::new(format!("offer: {}", min.name())).color(min.color()),
                    )
                    .show_ui(ui, |ui| {
                        for s in editor::NodeStatus::ALL.iter().rev() {
                            ui.selectable_value(
                                &mut min,
                                *s,
                                egui::RichText::new(s.filter_label()).color(s.color()),
                            );
                        }
                    })
                    .response
                    .on_hover_text(
                        "the least mature node the add menus offer. Every node is \
                         labelled production, beta or alpha; the graph marks beta and \
                         alpha with a small dot",
                    );
                if min != self.egraph.node_status_min {
                    self.egraph.node_status_min = min;
                }
            });
        self.globals_ui_open = open;
    }

    /// First launch: offer a cache directory on the system drive (the
    /// user's home), unless one is set already.
    pub(crate) fn cache_setup_begin(&mut self) {
        self.cache.setup_open = true;
    }

    /// The first-run cache window: the message, the drive's free space and
    /// the cache settings themselves. Returns true when it is closed.
    pub(crate) fn cache_setup_window(&mut self, ctx: &egui::Context) -> bool {
        if self.prefs.cache_dir.is_none() {
            self.prefs.cache_dir = default_cache_dir().map(|p| p.to_string_lossy().into_owned());
        }
        let mut done = false;
        egui::Window::new("ssd cache")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(460.0)
            .show(ctx, |ui| {
                ui.set_max_width(460.0);
                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new(
                        "Important! Wobble relies on an SSD cache for best performance. \
                         Please select a cache location.",
                    )
                    .size(15.0)
                    .strong(),
                );
                ui.add_space(8.0);
                let dir = self.prefs.cache_dir.clone().unwrap_or_default();
                let budget = self.prefs.disk_budget_gb.unwrap_or(64.0);
                match (!dir.trim().is_empty()).then(|| free_space(&engine::paths::expand_home(&dir))).flatten() {
                    Some((free, total)) => {
                        let (free_gb, total_gb) = (free as f64 / 1e9, total as f64 / 1e9);
                        let text = format!("that drive: {free_gb:.0} GB free of {total_gb:.0} GB");
                        if free_gb < budget || free_gb < total_gb * 0.1 {
                            ui.colored_label(
                                theme::warn(),
                                format!(
                                    "{text} — nearly full. pick another drive, or make the budget \
                                     smaller than what is free"
                                ),
                            );
                        } else {
                            ui.weak(text);
                        }
                    }
                    None if dir.trim().is_empty() => {
                        ui.colored_label(theme::warn(), "no cache: playback decodes every frame, every time");
                    }
                    None => {
                        ui.weak("free space unknown for that location");
                    }
                }
                ui.separator();
                self.cache_settings_ui(ui);
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button(egui::RichText::new("use this").strong()).clicked() {
                        self.apply_cache_dir();
                        done = self.cache.dir_err.is_none();
                    }
                    if ui.button("later").on_hover_text("no SSD cache for now — prefs ➡ cache sets one").clicked() {
                        // The suggestion was never applied: leave no trace.
                        if self.disk_cache.is_none() {
                            self.prefs.cache_dir = None;
                        }
                        done = true;
                    }
                    if let Some(e) = &self.cache.dir_err {
                        ui.colored_label(theme::error(), e);
                    }
                });
            });
        if done {
            self.cache.setup_open = false;
        }
        done
    }
}

/// `~/.wobble/cache`: on the system drive, which on a workstation is the
/// SSD more often than not. The window says how full it is either way.
fn default_cache_dir() -> Option<std::path::PathBuf> {
    crate::home_dir().map(|h| h.join(".wobble").join("cache"))
}

/// (free, total) bytes on the volume holding `path`, or its nearest
/// existing ancestor (the cache directory may not exist yet).
fn free_space(path: &std::path::Path) -> Option<(u64, u64)> {
    let mut p = path;
    while !p.exists() {
        p = p.parent()?;
    }
    volume_space(p)
}

#[cfg(unix)]
fn volume_space(p: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(p.as_os_str().as_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: a valid C string and a zeroed out-struct of the right type.
    if unsafe { libc::statvfs(c.as_ptr(), &mut s) } != 0 {
        return None;
    }
    let frsize = s.f_frsize as u64;
    Some((s.f_bavail as u64 * frsize, s.f_blocks as u64 * frsize))
}

#[cfg(windows)]
fn volume_space(p: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let w: Vec<u16> = p.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
    // SAFETY: a NUL-terminated wide path and three valid out-pointers.
    let ok = unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut avail, &mut total, &mut free) };
    (ok != 0).then_some((avail, total))
}

#[cfg(not(any(unix, windows)))]
fn volume_space(_p: &std::path::Path) -> Option<(u64, u64)> {
    None
}

impl Gfx {
    /// The prefs row for the scripting API: the switch (read at launch) and
    /// whether it is listening now. Turning it on opens it at once; off
    /// takes effect on the next launch (the listener has no close).
    fn api_settings_ui(&mut self, ui: &mut egui::Ui) {
        let mut on = self.prefs.scripting_api.unwrap_or(true);
        if ui
            .checkbox(&mut on, "open at launch")
            .on_hover_text(
                "lets scripts, addons and agents (wobble-mcp) drive this wobble\n\
                 loopback only (127.0.0.1); nothing on the network can reach it",
            )
            .changed()
        {
            self.prefs.scripting_api = Some(on);
            save_prefs(&self.prefs);
            if on && self.api.is_none() {
                self.api = crate::apiserve::ApiServe::start(api::port_from_env());
            }
        }
        let status = match (&self.api, on) {
            (Some(a), true) => format!("listening on 127.0.0.1:{}", a.port()),
            (Some(a), false) => format!("listening on 127.0.0.1:{} until restart", a.port()),
            (None, true) => format!("not listening: port {} is taken (another wobble?)", api::port_from_env()),
            (None, false) => "off".to_string(),
        };
        ui.label(egui::RichText::new(status).color(theme::text_weak()));
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn free_space_reads_the_nearest_existing_ancestor() {
        let missing = std::env::temp_dir().join("wobble-no-such-dir").join("deeper");
        let (free, total) = super::free_space(&missing).expect("the temp volume answers");
        assert!(total > 0 && free <= total);
    }
}
