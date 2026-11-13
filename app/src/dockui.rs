//! Dockable panels: imfram's dock (`imfram::dock`). The main window is a
//! host of it (an "external" one: wobble runs that window's egui frame
//! itself and draws the dock into the room under the menubar and the
//! timeline); a tab dragged out of it, or detached from its menu, gets a
//! window of imfram's own. Drag a tab and the leaves glide to where it
//! would land. Q / W (`viewer_mode`) show one leaf alone.
//!
//! The tabs: the node graph, the properties, the animation (curves and
//! dope sheet) and the viewer. EDIT MODE's comp window stays what it was
//! (`PaneKind::Comp`, `DetachedPane`): it is a mode, not a panel.

use egui::Ui;
// In a page the dock is a fixed layout of the same panes (`webdock`).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use imfram::dock;
#[cfg(target_arch = "wasm32")]
pub(crate) use crate::webdock as dock;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use imfram::Gpu;
#[cfg(target_arch = "wasm32")]
pub(crate) use crate::webdock::Gpu;
use dock::{Docks, Panels, Side, Tree};

use crate::{keymap, theme, Gfx, PaneKind};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum Pane {
    Nodes,
    Properties,
    Curves,
    Viewer,
    Timeline,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Role {
    Main,
    Detached,
}

impl Pane {
    /// The pane of the old scheme this tab is, for what still asks in
    /// those terms (`Gfx::detached`, the detach button).
    pub fn of(kind: PaneKind) -> Option<Pane> {
        match kind {
            PaneKind::Nodes => Some(Pane::Nodes),
            PaneKind::Properties => Some(Pane::Properties),
            PaneKind::Animation => Some(Pane::Curves),
            PaneKind::Comp => None,
        }
    }
}

impl dock::Tab for Pane {
    type Role = Role;
    fn all() -> &'static [Self] {
        &[Pane::Nodes, Pane::Properties, Pane::Curves, Pane::Viewer, Pane::Timeline]
    }
    fn title(self) -> &'static str {
        match self {
            Pane::Nodes => "nodes",
            Pane::Properties => "properties",
            Pane::Curves => "animation",
            Pane::Viewer => "viewer",
            Pane::Timeline => "timeline",
        }
    }
    fn help(self) -> &'static str {
        match self {
            Pane::Nodes => "the node graph",
            Pane::Properties => "the selected nodes' parameters",
            Pane::Curves => "keyframe curves and the dope sheet for the selected nodes",
            Pane::Viewer => "the image",
            Pane::Timeline => "the transport and the scrub track",
        }
    }
    fn home(self) -> Role {
        Role::Main
    }
    // The transport is a fixed two-row strip: taller only leaves room
    // under it, shorter cuts its controls off.
    fn strip(self) -> Option<f32> {
        (self == Pane::Timeline).then_some(crate::TIMELINE_H)
    }
}

impl dock::Role for Role {
    fn title(self) -> &'static str {
        "wobble"
    }
    fn size(self) -> (u32, u32) {
        (900, 600)
    }
    fn detached() -> Self {
        Role::Detached
    }
}

/// The main window's layout at first: the graph left of the viewer, the
/// timeline under both, the properties (with the animation beside it)
/// along the bottom.
pub fn layout() -> Tree<Pane> {
    let mut t = Tree::new(vec![Pane::Viewer]);
    t.split(Side::Left, 0.6, vec![Pane::Nodes]);
    t.split(Side::Bottom, 0.85, vec![Pane::Timeline]);
    t.split(Side::Bottom, 0.7, vec![Pane::Properties, Pane::Curves]);
    t
}

/// A layout saved before the timeline was a tab: it goes above the
/// properties (where it sat before the dock), else under the viewer.
pub fn add_timeline(docks: &mut WobbleDocks) {
    if docks.host_of(Pane::Timeline).is_none() {
        add_timeline_to(&mut docks.hosts[MAIN].tree);
    }
}

fn add_timeline_to(tree: &mut Tree<Pane>) {
    use dock::{Drop, Zone};
    let drop = match (tree.find(Pane::Properties), tree.find(Pane::Viewer)) {
        (Some(l), _) => Drop::Leaf(l, Zone::Side(Side::Top)),
        (None, Some(l)) => Drop::Leaf(l, Zone::Side(Side::Bottom)),
        _ => Drop::Root(Side::Bottom),
    };
    tree.insert(Pane::Timeline, drop);
}

/// The docks of wobble: the main window first (`MAIN`), then any windows
/// tabs were detached into.
pub type WobbleDocks = Docks<Pane>;
pub const MAIN: usize = 0;

/// Our own windows clear to the field colour, as the panes did.
pub fn clear_color() -> wgpu::Color {
    let c = theme::field();
    wgpu::Color { r: c.r() as f64 / 255.0, g: c.g() as f64 / 255.0, b: c.b() as f64 / 255.0, a: 1.0 }
}

impl Panels<Pane> for Gfx {
    // (A detached window's frame: the same Tab steal as the main window's,
    // so the search opens in whichever graph is hovered.)
    fn raw_input(&mut self, ctx: &egui::Context, raw: &mut egui::RawInput, _tabs: &[Pane]) {
        let tab = keymap::take_search(ctx, raw);
        self.node_editor.tab_open_request = tab;
        self.patch_ui.tab_request = tab;
    }

    fn begin(&mut self, ui: &mut Ui) {
        theme::sync(ui.ctx());
    }

    // (The main window has its menubar; a detached window's tabs stand
    // alone. Closed tabs come back from the view menu.)
    fn bar_shown(&self) -> bool {
        false
    }

    fn tab(&mut self, ui: &mut Ui, pane: Pane) {
        self.tab_body(ui, pane);
    }

    // The picture is blitted to the surface before egui draws; a filled
    // body would paint over it.
    fn see_through(&self, tab: &Pane) -> bool {
        *tab == Pane::Viewer
    }

    fn scroll_bars(&self, _tab: &Pane) -> [bool; 2] {
        [false, false]
    }

    // No row of tab headers: each pane's first row of controls starts
    // with a grip (`dock::grip`) that drags, stacks and has the tab menu.
    fn tab_bars(&self) -> bool {
        false
    }
}

impl Gfx {
    fn tab_body(&mut self, ui: &mut Ui, pane: Pane) {
        match pane {
            Pane::Nodes => {
                ui.horizontal(|ui| {
                    dock::grip(ui);
                    self.dive_bar_ui(ui);
                    ui.separator();
                    self.graph_bar_items(ui);
                });
                self.graph_panel_rect = ui.max_rect();
                self.pane_ui(ui, PaneKind::Nodes);
            }
            Pane::Properties => {
                self.bottom = crate::BottomTab::Properties;
                self.bottom_applied = self.bottom;
                self.bottom_rect = ui.max_rect();
                if self.dock_drawing_main {
                    ui.horizontal(|ui| {
                        dock::grip(ui);
                        self.prop_layout_bar(ui);
                        self.props_keys_strip(ui);
                    });
                    ui.separator();
                    self.properties_ui(ui, false);
                } else {
                    self.pane_ui(ui, PaneKind::Properties);
                }
            }
            Pane::Curves => {
                self.bottom = crate::BottomTab::Curves;
                self.bottom_applied = self.bottom;
                self.bottom_rect = ui.max_rect();
                ui.horizontal(|ui| {
                    dock::grip(ui);
                    ui.weak(if self.dope_view { "key timing for the selected nodes" } else { "keyframe curves for the selected nodes" });
                });
                ui.separator();
                self.animation_ui(ui);
            }
            Pane::Timeline => {
                let r = ui.available_rect_before_wrap();
                let t = egui::Rect::from_min_size(r.min, egui::vec2(r.width(), crate::TIMELINE_H));
                ui.scope_builder(egui::UiBuilder::new().max_rect(t), |ui| self.timeline_ui(ui));
            }
            Pane::Viewer => {
                // A bar wider than a narrow pane widens the ui it is in;
                // the picture still only gets the pane (it was framed off
                // the widened rect, half outside the viewer, in a page).
                let pane = ui.max_rect();
                self.viewer_bar_ui(ui);
                let r = ui.available_rect_before_wrap().intersect(pane);
                ui.scope_builder(egui::UiBuilder::new().max_rect(r), |ui| self.viewer_ui(ui));
            }
        }
    }}

#[cfg(test)]
mod tests {
    use super::*;
    use imfram::dock::Tab;

    #[test]
    fn an_old_layout_gets_the_timeline_full_width_above_the_properties() {
        // The shape a pre-timeline prefs file holds: graph | viewer over
        // the properties.
        let mut t = Tree::new(vec![Pane::Viewer]);
        t.split(Side::Left, 0.5, vec![Pane::Nodes, Pane::Curves]);
        t.split(Side::Bottom, 0.72, vec![Pane::Properties]);
        add_timeline_to(&mut t);
        let fixed = |tabs: &[Pane]| tabs.iter().all(|p| p.strip().is_some()).then_some(crate::TIMELINE_H);
        let l = t.layout_fixed(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(1000.0, 800.0)), 4.0, &fixed);
        let rect = |p: Pane| l.leaves.iter().find(|(id, _)| Some(*id) == t.find(p)).unwrap().1;
        let (tl, props, viewer) = (rect(Pane::Timeline), rect(Pane::Properties), rect(Pane::Viewer));
        assert_eq!(tl.width(), 1000.0, "full width");
        assert!(tl.max.y <= props.min.y && tl.min.y >= viewer.max.y, "between the viewer and the properties");
        assert!((tl.height() - crate::TIMELINE_H).abs() < 0.01);
    }
}
