//! The load-clip UI: seq_manip's browser (`~/src/devl/seq_manip_imfram`,
//! its `seq` library in embedded mode, branch `edify-embed`) — the same
//! grid / list, thumbnails, path bar, filters, subfolder search, type-ahead
//! and staging verbs as the standalone app, on its prefs.
//!
//! Presented as wobble presents its clip browser: a big window over the app
//! (wobble's fills the window; here it fills everything ABOVE the timeline,
//! so clips can be dragged straight down onto it — it moves and resizes
//! like any window), opened by cmd+I, file > import and the bin's
//! "browse…" button, closed by its × or Esc.
//!
//! - Enter / double-click on a sequence or movie: it joins the bin and loads
//!   into the source viewer, and the window closes (a pick, as wobble's
//!   single pick for a Load).
//! - The window is seq_manip's dock flattened (`seq::embed`): folders
//!   (favourites + tree) down the left, the browser, Info / Pending /
//!   Delete queue down the right; widths kept in seq_manip's prefs.
//! - The TO IMPORT tray (seq_manip's bin, foot of the left side, where
//!   wobble keeps its collect-into-bin tray): B, "add to import" or a drag
//!   collects picks across folders; Backspace / Remove, ⏶ ⏷, Clear;
//!   "Import (n)" puts them in the bin in the tray's order and closes.
//!   Dragged from the tray to the timeline they lay down in that order.
//! - Drag rows out of the window onto the timeline (overwrite there, cmd:
//!   insert) or the bin: every dragged clip, in order. Inside the window a
//!   drag still moves files into a folder, as in seq_manip.
//! - Inside an organisicator project a row of shortcuts (project, shot, the
//!   shot's folders) sits over the browser.
//! - While the pointer is over it Scrub's own keys stand down.
//!
//! Folder listing is seq_manip's own synchronous call (wobble's browser
//! lists synchronously too); thumbnails and metadata come off-thread.
//!
//! The `clips` feature (on by default) builds it; without it the bin has no
//! browse button.

use crate::editor::Editor;
use crate::timelineui::Payload;
use egui::Rect;
use imfram::gallery::Dragged;
use std::path::PathBuf;

pub struct ClipBrowser {
    pub seq: seq::app::Seq,
    pub open: bool,
    /// The window's rect last frame (keys and drags ask it).
    pub rect: Option<Rect>,
    /// The bin's folder last opened from (a new one — a post's rushes —
    /// takes the browser there).
    from: Option<PathBuf>,
    /// The organisicator project of the folder shown, by folder.
    org: Option<(PathBuf, Option<crate::organisicator::Project>)>,
    /// Press "Import (n)" next frame (tests).
    import_now: bool,
}

impl ClipBrowser {
    pub fn new(headless: bool) -> Self {
        let mut s = seq::app::Seq::new(None);
        s.embedded = true;
        s.ask = None; // no first-run splash inside Scrub
        s.keep_prefs = headless;
        ClipBrowser { seq: s, open: false, rect: None, from: None, org: None, import_now: false }
    }
}

impl Editor {
    /// cmd+I / file > import / the bin's browse button: open the browser
    /// (at the bin's folder when that changed since it was last opened).
    pub(crate) fn open_clips(&mut self) {
        let dir = self.bin.dir.clone();
        let Some(cb) = &mut self.clips else { return };
        if cb.from.as_ref() != Some(&dir) && dir.is_dir() {
            cb.seq.go(&dir);
            cb.from = Some(dir);
        }
        cb.open = true;
    }

    /// The pointer is over the open browser window (Scrub's keys stand down).
    pub(crate) fn over_clips(&self, ui: &egui::Ui) -> bool {
        let Some(cb) = self.clips.as_ref().filter(|c| c.open) else { return false };
        let q = ui.input(|i| i.pointer.hover_pos());
        cb.rect.zip(q).is_some_and(|(r, q)| r.contains(q))
    }

    /// The browser window, and what it picked.
    pub(crate) fn clips_window(&mut self, ctx: &egui::Context) {
        let area = ctx.content_rect();
        let area = match self.panes.rect_of(crate::panes::Pane::Timeline) {
            Some(t) if t.min.y > area.min.y + 200.0 => Rect::from_min_max(egui::pos2(area.min.x, area.min.y + 24.0), egui::pos2(area.max.x, t.min.y - 4.0)),
            _ => area,
        };
        let Some(cb) = &mut self.clips else { return };
        if !cb.open {
            cb.rect = None;
            return;
        }
        cb.seq.jobs.set_ctx(ctx);
        if cb.seq.poll() {
            ctx.request_repaint();
        }
        // Organisicator shortcuts for the folder shown.
        if cb.org.as_ref().is_none_or(|(d, _)| *d != cb.seq.dir) {
            cb.org = Some((cb.seq.dir.clone(), crate::organisicator::find(&cb.seq.dir)));
        }
        let mut open = true;
        let mut go: Option<PathBuf> = None;
        let (mut to_tray, mut to_source, mut import) = (false, false, false);
        let resp = egui::Window::new("clips — Enter loads into the source, B collects to import")
            .open(&mut open)
            .collapsible(false)
            .default_rect(area)
            // Up at once (a fade-in leaves it see-through over the viewers
            // for its first frames).
            .fade_in(false)
            .show(ctx, |ui| {
                if let Some((_, Some(p))) = &cb.org {
                    let inst = p.locate(&cb.seq.dir);
                    ui.horizontal_wrapped(|ui| {
                        ui.weak("organisicator");
                        if ui.small_button(p.label(inst)).on_hover_text("the project root").clicked() {
                            go = Some(p.root.clone());
                        }
                        if let Some(i) = inst {
                            if ui.small_button("shot").clicked() {
                                go = Some(p.abs(&i.rel));
                            }
                        }
                        for (f, path) in p.folders_of(inst) {
                            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                            if path.is_dir() && ui.small_button(name).on_hover_text(f.hint.clone()).clicked() {
                                go = Some(path);
                            }
                        }
                    });
                }
                egui::Panel::bottom("scrub-clips-foot").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let n = cb.seq.selected().iter().filter(|r| seq::media_path(r).is_some()).count();
                        to_tray = ui.add_enabled(n > 0, egui::Button::new(format!("add to import ({n})  B"))).on_hover_text("collect into the to-import tray (bottom left); it keeps them while you browse").clicked();
                        to_source = ui.add_enabled(n > 0, egui::Button::new("open in source  (Enter)")).clicked();
                        ui.weak("drag clips onto the timeline or the bin; Esc closes");
                    });
                });
                import = seq::embed::show(ui, &mut cb.seq).import;
            });
        cb.rect = resp.map(|r| r.response.rect);
        // The app's dialogs and notes (rename, asks, progress, toasts).
        seq::verbs::show(ctx, &mut cb.seq);
        seq::panels::asks(ctx, &mut cb.seq);
        seq::panels::progress(ctx, &mut cb.seq);
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        cb.seq.toasts.tick(dt);
        cb.seq.toasts.show(ctx);
        if !cb.seq.toasts.is_empty() {
            ctx.request_repaint();
        }
        cb.seq.dock_asks.clear();
        if let Some(d) = go {
            cb.seq.go(&d);
        }
        let typing = ctx.egui_wants_keyboard_input();
        if !open || (!typing && cb.seq.verb.is_none() && ctx.input(|i| i.key_pressed(egui::Key::Escape))) {
            cb.open = false;
        }
        // Rows dragged out of the window become files Scrub understands
        // (every media row of the drag, in order).
        let inside = ctx.input(|i| i.pointer.hover_pos()).zip(cb.rect).is_some_and(|(q, r)| r.contains(q));
        if !inside {
            if let Some(d) = egui::DragAndDrop::payload::<Dragged>(ctx) {
                let paths: Vec<PathBuf> = d.0.iter().filter_map(|id| cb.seq.row(*id).and_then(seq::media_path)).collect();
                match paths.len() {
                    0 => {}
                    1 => egui::DragAndDrop::set_payload(ctx, Payload::File(paths[0].clone())),
                    _ => egui::DragAndDrop::set_payload(ctx, Payload::Files(paths)),
                }
            }
        }
        let selected: Vec<PathBuf> = cb.seq.selected().iter().filter_map(|r| seq::media_path(r)).collect();
        let opened: Vec<PathBuf> = cb.seq.opened.drain(..).collect();
        if to_tray {
            let ids = cb.seq.selected_ids();
            seq::embed::add_to_tray(&mut cb.seq, ids);
        }
        // The tray's "Import (n)": into the bin in the tray's order.
        let import = import || std::mem::take(&mut cb.import_now);
        let imported: Vec<PathBuf> = if import {
            let paths: Vec<PathBuf> = seq::embed::tray_rows(&cb.seq).iter().filter_map(seq::media_path).collect();
            seq::embed::clear_tray(&mut cb.seq);
            cb.open = false;
            paths
        } else {
            vec![]
        };
        // A pick: into the bin and the source, and the window closes.
        if let Some(p) = opened.last().cloned().or_else(|| to_source.then(|| selected.first().cloned()).flatten()) {
            cb.open = false;
            for q in opened.iter().chain(std::iter::once(&p)) {
                self.bin.add(q, String::new());
            }
            self.open(&p);
            self.say(format!("source: {}", p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()));
        }
        if import {
            for p in &imported {
                self.bin.add(p, String::new());
            }
            self.say(format!("{} clip(s) imported into the bin", imported.len()));
        }
    }

    /// The bin takes files dropped on it (from the browser).
    pub(crate) fn bin_drop(&mut self, ui: &egui::Ui, bin: Rect) {
        let released = ui.input(|i| i.pointer.any_released() && i.pointer.hover_pos().is_some_and(|q| bin.contains(q)));
        if !released {
            return;
        }
        let Some(pl) = egui::DragAndDrop::take_payload::<Payload>(ui.ctx()) else { return };
        let keys: Vec<crate::bin::ItemKey> = match &*pl {
            Payload::File(p) => vec![(p.clone(), None)],
            Payload::Files(v) => v.iter().map(|p| (p.clone(), None)).collect(),
            Payload::Clips(v) => v.clone(),
            _ => return,
        };
        // Onto a bin's name: into that bin (moved there if already in).
        let q = ui.input(|i| i.pointer.hover_pos());
        let head = q.and_then(|q| self.bin.header_rects.iter().find(|(_, r)| r.contains(q)).map(|(g, _)| g.clone()));
        if let Some(g) = head {
            if self.bin.takes_moves(&g) {
                let n = self.bin.drop_into(&keys, &g);
                self.say(format!("{n} clip(s) into {}", if g.is_empty() { "clips" } else { &g }));
            } else {
                self.say("drop clips onto \"clips\" or a bin you made (N makes one)");
            }
            return;
        }
        // A drag that started in the bin and ended in it: nothing to add.
        if keys.iter().all(|(p, r)| r.is_some() || self.bin.clip(p).is_some()) {
            return;
        }
        let paths: Vec<PathBuf> = keys.into_iter().filter(|(_, r)| r.is_none()).map(|(p, _)| p).collect();
        for p in &paths {
            self.bin.add(p, String::new());
        }
        self.say(format!("{} clip(s) added to the bin", paths.len()));
    }

    /// Go to a folder in the browser (tests, the API later).
    pub fn clips_go(&mut self, dir: &std::path::Path) -> bool {
        match &mut self.clips {
            Some(cb) => cb.seq.go(dir),
            None => return false,
        }
        true
    }

    pub fn clips_open(&self) -> bool {
        self.clips.as_ref().is_some_and(|c| c.open)
    }

    pub fn clips_rect(&self) -> Option<Rect> {
        self.clips.as_ref().and_then(|c| c.rect)
    }

    /// The browser's rows: (id, media path) for media (tests).
    pub fn clips_rows(&self) -> Vec<(u64, PathBuf)> {
        self.clips.as_ref().map_or_else(Vec::new, |cb| cb.seq.rows.iter().filter_map(|r| seq::media_path(r).map(|p| (r.id, p))).collect())
    }

    /// Select rows in the browser (tests).
    pub fn clips_select(&mut self, ids: &[u64]) {
        if let Some(cb) = &mut self.clips {
            cb.seq.selection = ids.iter().copied().collect();
        }
    }

    /// The to-import tray's media, in order (tests).
    pub fn clips_tray(&self) -> Vec<PathBuf> {
        self.clips.as_ref().map_or_else(Vec::new, |cb| seq::embed::tray_rows(&cb.seq).iter().filter_map(seq::media_path).collect())
    }

    /// Press the tray's "Import (n)" (tests).
    pub fn clips_import(&mut self) {
        if let Some(cb) = &mut self.clips {
            cb.import_now = true;
        }
    }

    /// Activate a row as Enter would (tests).
    pub fn clips_activate(&mut self, id: u64) {
        if let Some(cb) = &mut self.clips {
            seq::browser::open(&mut cb.seq, id);
        }
    }
}
