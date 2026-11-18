//! The window's panels on imfram's dock (`imfram::dock`), as wobble and
//! graav have them: each panel (bin, source, record, timeline) is a tab in
//! a split tree. Drag a panel by its grip (the small patch at the start of
//! its top row) and the leaves glide to show where it would land; let go
//! to put it there. Dragged out of the window it becomes a window of its
//! own (a real OS window), or docks into another Scrub window it is let go
//! over; a grip's right-click menu detaches or moves it too. Closing a
//! torn-off window puts its panels back.
//!
//! - **Q** over a panel shows its leaf ALONE (the rest fold away, gliding);
//!   Q again (or Esc) puts the layout back. The dock's own solo.
//! - **W** over a panel closes it (the rest take its room); the view menu,
//!   or shift+W for all, brings it back where it was.
//! - window > reset layout puts the default back.
//!
//! The main window is the dock's EXTERNAL host: Scrub runs its egui frame
//! (menubar, per-frame transport work, the floating windows) and hands the
//! room under the menubar to `Docks::draw_external`; torn-off windows are
//! imfram's own, each with its own viewer GPU state (`viewer::Viewers`).
//!
//! The layout, the key-hint and readout switches and the bin's view are
//! kept in `~/Library/Application Support/Scrub/ui.json` (not in headless
//! runs). A ui.json from before the dock gives its bin width, timeline
//! height and viewer split to the first layout.

use egui::{Pos2, Rect};
use imfram::dock::{self, Side, Tree};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pane {
    Bin,
    Source,
    Record,
    Timeline,
    /// EXPERIMENT (the `fold` feature): wobble itself, hosted (fold.rs).
    Wobble,
    /// EXPERIMENT (fold): the controls of the wob clip under the playhead.
    Controls,
}

#[cfg(not(feature = "fold"))]
pub const ALL: [Pane; 4] = [Pane::Bin, Pane::Source, Pane::Record, Pane::Timeline];
#[cfg(feature = "fold")]
pub const ALL: [Pane; 6] = [Pane::Bin, Pane::Source, Pane::Record, Pane::Timeline, Pane::Wobble, Pane::Controls];

impl Pane {
    pub fn name(self) -> &'static str {
        match self {
            Pane::Bin => "bin",
            Pane::Source => "source",
            Pane::Record => "record",
            Pane::Timeline => "timeline",
            Pane::Wobble => "wobble",
            Pane::Controls => "wob controls",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    Main,
    Detached,
}

impl dock::Tab for Pane {
    type Role = Role;
    fn all() -> &'static [Self] {
        &ALL
    }
    fn title(self) -> &'static str {
        self.name()
    }
    fn help(self) -> &'static str {
        match self {
            Pane::Bin => "the clips brought in (list or scrub tiles)",
            Pane::Source => "the source viewer: the clip loaded, its in and out",
            Pane::Record => "the record viewer: the sequence",
            Pane::Timeline => "the sequence's tracks",
            Pane::Wobble => "wobble, hosted: the comp of the wob clip under the playhead",
            Pane::Controls => "the controls the wob clip under the playhead publishes",
        }
    }
    fn home(self) -> Role {
        Role::Main
    }
}

impl dock::Role for Role {
    fn title(self) -> &'static str {
        "Scrub"
    }
    fn size(self) -> (u32, u32) {
        (960, 600)
    }
    fn detached() -> Self {
        Role::Detached
    }
}

pub type Docks = dock::Docks<Pane>;
pub type Saved = dock::Layout<Pane, Role>;
pub const MAIN: usize = 0;

const BIN_W: f32 = 300.0;
const TIMELINE_H: f32 = 210.0;

/// The layout at first (and window > reset layout): the bin down the left,
/// the timeline along the bottom, source and record side by side over it.
/// Sizes from a pre-dock ui.json when there is one (fractions of a
/// 1600x1000 window).
pub fn default_tree(bin_w: f32, timeline_h: f32, split: f32) -> Tree<Pane> {
    let mut t = Tree::new(vec![Pane::Source]);
    t.split(Side::Right, split.clamp(0.15, 0.85), vec![Pane::Record]);
    t.split(Side::Bottom, (1.0 - timeline_h / 975.0).clamp(0.3, 0.9), vec![Pane::Timeline]);
    t.split(Side::Left, (1.0 - bin_w / 1600.0).clamp(0.4, 0.95), vec![Pane::Bin]);
    t
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Panes {
    /// Pre-dock sizes, read once to seed the first layout.
    pub bin_w: f32,
    pub timeline_h: f32,
    pub split: f32,
    /// The windows and their split trees as last left.
    pub dock: Option<Saved>,
    /// The key-hint strip over the viewers (H).
    pub hints: bool,
    /// The viewers' readouts: frame, decode, cache… (shift+H).
    pub readouts: bool,
    /// Experimental: Wobble clips (only offered in a `wobble` build).
    pub experimental_wobble: bool,
    /// The bin shows clips as hover-scrub tiles, and their width.
    pub bin_tiles: bool,
    pub bin_tile: f32,
    /// The screen recorder's options (area, cursor, mark, folders).
    pub rec_opts: Option<crate::recorder::Opts>,
    /// The first-launch hotkey tour has been shown (and closed).
    pub tour_seen: bool,
    /// Where each panel was drawn last frame, by window.
    #[serde(skip)]
    last: Vec<(usize, Pane, Rect)>,
    /// This frame's, as the panels draw.
    #[serde(skip)]
    now: Vec<(usize, Pane, Rect)>,
    /// The window being drawn.
    #[serde(skip)]
    pub host: usize,
    /// Open and soloed, as the dock had them at the end of last frame.
    #[serde(skip)]
    pub open: Vec<Pane>,
    #[serde(skip)]
    pub solo: Option<Pane>,
    #[serde(skip)]
    headless: bool,
    #[serde(skip)]
    dirty: bool,
    /// The layout as last written.
    #[serde(skip)]
    saved_dock: String,
}

impl Default for Panes {
    fn default() -> Self {
        Self {
            bin_w: BIN_W,
            timeline_h: TIMELINE_H,
            split: 0.5,
            dock: None,
            hints: true,
            readouts: true,
            experimental_wobble: false,
            bin_tiles: false,
            bin_tile: 160.0,
            rec_opts: None,
            tour_seen: false,
            last: Vec::new(),
            now: Vec::new(),
            host: MAIN,
            open: ALL.to_vec(),
            solo: None,
            headless: false,
            dirty: false,
            saved_dock: String::new(),
        }
    }
}

fn path() -> PathBuf {
    media::sys::support_dir().join("ui.json")
}

impl Panes {
    pub fn load(headless: bool) -> Self {
        let mut p: Panes = if headless { Panes::default() } else { std::fs::read(path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default() };
        p.headless = headless;
        p
    }

    fn save(&mut self) {
        self.dirty = false;
        if self.headless {
            return;
        }
        if let Ok(b) = serde_json::to_vec_pretty(self) {
            let _ = std::fs::create_dir_all(path().parent().expect("a parent"));
            let _ = crate::project::write_atomic(&path(), &b);
        }
    }

    /// The default tree (seeded from pre-dock sizes).
    pub fn default_tree(&self) -> Tree<Pane> {
        default_tree(self.bin_w, self.timeline_h, self.split)
    }

    /// The layout as last kept, if any (taken: restored once).
    pub fn take_saved(&mut self) -> Option<Saved> {
        self.dock.take()
    }

    /// Kept once the pointer lets go, if the layout changed.
    pub fn note_layout(&mut self, docks: &Docks, released: bool) {
        if !released || self.headless {
            return;
        }
        let l = docks.save();
        let s = serde_json::to_string(&l).unwrap_or_default();
        if s != self.saved_dock {
            self.saved_dock = s;
            self.dock = Some(l);
            self.dirty = true;
        }
        if self.dirty {
            self.save();
        }
    }

    /// Something kept in ui.json changed (saved once the pointer lets go).
    pub fn changed(&mut self) {
        self.dirty = true;
    }

    /// A panel drew itself at `r` in the window being drawn.
    pub fn drew(&mut self, p: Pane, r: Rect) {
        let h = self.host;
        self.now.retain(|(hh, q, _)| !(*hh == h && *q == p));
        self.now.push((h, p, r));
    }

    /// Window `host` starts a frame.
    pub fn begin_host(&mut self, host: usize) {
        self.host = host;
        self.now.retain(|(h, _, _)| *h != host);
    }

    /// Window `host`'s frame is over: its rects are this frame's.
    pub fn end_host(&mut self, host: usize) {
        let mine: Vec<_> = self.now.iter().filter(|(h, _, _)| *h == host).cloned().collect();
        self.last.retain(|(h, _, _)| *h != host);
        self.last.extend(mine);
    }

    /// A window went away: forget its rects.
    pub fn forget_hosts_from(&mut self, n: usize) {
        self.last.retain(|(h, _, _)| *h < n);
        self.now.retain(|(h, _, _)| *h < n);
    }

    /// The panel under `at` in the window being drawn (last frame's rects);
    /// `false`: not a restore strip (there are none on the dock).
    pub fn at(&self, at: Pos2) -> Option<(Pane, bool)> {
        self.last.iter().find(|(h, _, r)| *h == self.host && r.contains(at)).map(|(_, p, _)| (*p, false))
    }

    /// Where a panel was drawn last (the main window first; tests aim with it).
    pub fn rect_of(&self, p: Pane) -> Option<Rect> {
        let mut v: Vec<_> = self.last.iter().filter(|(_, q, _)| *q == p).collect();
        v.sort_by_key(|(h, _, _)| *h);
        v.first().map(|(_, _, r)| *r)
    }

    /// Panels drawn in window `host` last frame.
    pub fn in_host(&self, host: usize) -> Vec<(Pane, Rect)> {
        self.last.iter().filter(|(h, _, _)| *h == host).map(|(_, p, r)| (*p, *r)).collect()
    }

    pub fn shown(&self, p: Pane) -> bool {
        match self.solo {
            Some(s) => s == p,
            None => self.open.contains(&p),
        }
    }

    pub fn set_hints(&mut self, on: bool) {
        self.hints = on;
        self.save();
    }

    #[cfg_attr(not(feature = "wobble"), allow(dead_code))]
    pub fn set_experimental_wobble(&mut self, on: bool) {
        self.experimental_wobble = on;
        self.save();
    }

    pub fn set_tour_seen(&mut self) {
        self.tour_seen = true;
        self.save();
    }

    pub fn set_readouts(&mut self, on: bool) {
        self.readouts = on;
        self.save();
    }
}

/// shift+R: the split between the two viewers to half and half (when they
/// share one).
pub fn match_viewers(tree: &mut Tree<Pane>) {
    let area = Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1600.0, 1000.0));
    let lay = tree.layout(area, 6.0);
    let rect = |p: Pane| tree.find(p).and_then(|l| lay.leaves.iter().find(|(id, _)| *id == l)).map(|(_, r)| *r);
    let (Some(a), Some(b)) = (rect(Pane::Source), rect(Pane::Record)) else { return };
    // The smallest split holding both, with them on either side of its band.
    let apart = |band: Rect| (band.center().x - a.center().x) * (band.center().x - b.center().x) + (band.center().y - a.center().y) * (band.center().y - b.center().y) < 0.0;
    let best = lay.seps.iter().filter(|(_, band, whole, _)| whole.contains_rect(a) && whole.contains_rect(b) && apart(*band)).min_by(|x, y| x.2.area().total_cmp(&y.2.area()));
    if let Some((path, _, _, _)) = best {
        let path = path.clone();
        tree.set_frac(&path, 0.5);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_layout_has_every_panel_and_shift_r_evens_the_viewers() {
        let mut t = default_tree(300.0, 210.0, 0.7);
        // Wobble and its controls (the `fold` feature) open on demand.
        for p in ALL.into_iter().filter(|p| !matches!(p, Pane::Wobble | Pane::Controls)) {
            assert!(t.find(p).is_some(), "{p:?}");
        }
        let area = Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1600.0, 1000.0));
        let w = |t: &Tree<Pane>, p: Pane| {
            let lay = t.layout(area, 6.0);
            lay.leaves.iter().find(|(id, _)| Some(*id) == t.find(p)).map(|(_, r)| r.width()).unwrap()
        };
        assert!(w(&t, Pane::Source) > w(&t, Pane::Record) * 1.5);
        match_viewers(&mut t);
        assert!((w(&t, Pane::Source) - w(&t, Pane::Record)).abs() < 2.0);
        assert!(w(&t, Pane::Bin) > 250.0 && w(&t, Pane::Bin) < 350.0);
    }
}
