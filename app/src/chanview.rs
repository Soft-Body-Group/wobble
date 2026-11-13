//! Which animated channels the animation editors show — ONE state, read by
//! both the curve editor and the dope sheet, so the two always list the same
//! channels in the same order and a channel hidden in one is hidden in the
//! other.
//!
//! Listing: the selected nodes' animated channels (every node's when nothing
//! is selected) plus anything PINNED — unless a saved SET is active, in which
//! case the listing is exactly that set, whatever is selected. `hidden` then
//! filters the listing down to what is drawn.
//!
//! Sets are named channel collections saved IN THE SETUP (they name nodes by
//! id, so they belong to the comp, not to the app). "save" records the
//! channels currently VISIBLE; loading one makes it active.

use crate::anim::Channel;
use crate::editor::EditorGraph;
use std::collections::{BTreeSet, HashSet};

pub type ChanKey = (u64, Channel);

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ChanSet {
    pub name: String,
    pub chans: Vec<ChanKey>,
}

#[derive(Default)]
pub struct ChannelView {
    pub hidden: HashSet<ChanKey>,
    /// Kept in the listing when the node selection moves on.
    pub pinned: BTreeSet<ChanKey>,
    /// Saved collections (persisted with the setup).
    pub sets: Vec<ChanSet>,
    /// Index into `sets` while one is showing.
    pub active: Option<usize>,
    /// Name typed into the sets menu's "save as" field.
    draft: String,
}

impl ChannelView {
    /// Drop references to nodes that are gone; an active set that has lost
    /// its index is left.
    pub fn prune(&mut self, graph: &EditorGraph) {
        self.pinned.retain(|(n, _)| graph.nodes.contains_key(n));
        if self.active.is_some_and(|i| i >= self.sets.len()) {
            self.active = None;
        }
    }

    /// The listed channels, keyed and animated, in the order both editors
    /// draw them: by node id, then label (`label` is the editors' own
    /// "node · channel" text, so the sort matches what you read).
    pub fn listed(
        &self,
        graph: &EditorGraph,
        sel: &[u64],
        label: impl Fn(u64, Channel) -> String,
    ) -> Vec<ChanKey> {
        let mut chans: Vec<ChanKey> = Vec::new();
        if let Some(set) = self.active.and_then(|i| self.sets.get(i)) {
            chans.extend(set.chans.iter().copied());
        } else {
            let ids: Vec<u64> =
                if sel.is_empty() { graph.nodes.keys().copied().collect() } else { sel.to_vec() };
            for id in &ids {
                if let Some(n) = graph.nodes.get(id) {
                    chans.extend(n.anim.keys().map(|ch| (*id, *ch)));
                }
            }
            for p in &self.pinned {
                if !chans.contains(p) {
                    chans.push(*p);
                }
            }
        }
        chans.retain(|(n, ch)| {
            graph.nodes.get(n).and_then(|nd| nd.anim.get(ch)).is_some_and(|c| !c.keys.is_empty())
        });
        let mut keyed: Vec<(u64, String, ChanKey)> =
            chans.into_iter().map(|k| (k.0, label(k.0, k.1), k)).collect();
        keyed.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
        keyed.dedup_by(|a, b| a.2 == b.2);
        keyed.into_iter().map(|(.., k)| k).collect()
    }

    /// The listed channels that are drawn.
    pub fn visible(&self, listed: &[ChanKey]) -> Vec<ChanKey> {
        listed.iter().copied().filter(|k| !self.hidden.contains(k)).collect()
    }

    /// Save the visible channels as `name` (replacing a set of that name)
    /// and make it active. Returns false when there is nothing to save.
    pub fn save(&mut self, name: &str, listed: &[ChanKey]) -> bool {
        let chans = self.visible(listed);
        let name = name.trim();
        if chans.is_empty() || name.is_empty() {
            return false;
        }
        let set = ChanSet { name: name.to_string(), chans };
        let i = match self.sets.iter().position(|s| s.name == name) {
            Some(i) => {
                self.sets[i] = set;
                i
            }
            None => {
                self.sets.push(set);
                self.sets.len() - 1
            }
        };
        self.load(i);
        true
    }

    /// Show set `i`: it becomes the listing, every channel in it visible.
    pub fn load(&mut self, i: usize) {
        if let Some(s) = self.sets.get(i) {
            for k in &s.chans {
                self.hidden.remove(k);
            }
            self.active = Some(i);
        }
    }

    pub fn delete(&mut self, i: usize) {
        if i < self.sets.len() {
            self.sets.remove(i);
            self.active = match self.active {
                Some(a) if a == i => None,
                Some(a) if a > i => Some(a - 1),
                a => a,
            };
        }
    }

    /// The "sets" menu both editors put in their header: the saved sets
    /// (click shows one, click the active one to leave it), save-as, and
    /// per-set overwrite / delete.
    pub fn sets_menu(&mut self, ui: &mut egui::Ui, listed: &[ChanKey]) {
        let title = match self.active.and_then(|i| self.sets.get(i)) {
            Some(s) => format!("set: {}", s.name),
            None => "sets".to_string(),
        };
        ui.menu_button(egui::RichText::new(title).small(), |ui| {
            ui.set_min_width(200.0);
            if self.sets.is_empty() {
                ui.weak("no saved sets yet");
            }
            let mut load = None;
            let mut overwrite = None;
            let mut delete = None;
            for (i, s) in self.sets.iter().enumerate() {
                ui.horizontal(|ui| {
                    let on = self.active == Some(i);
                    if ui
                        .selectable_label(on, format!("{} ({})", s.name, s.chans.len()))
                        .on_hover_text(if on {
                            "showing — click to go back to the selection"
                        } else {
                            "show exactly these channels, whatever is selected"
                        })
                        .clicked()
                    {
                        load = Some(i);
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.small_button("×").on_hover_text("delete this set").clicked() {
                            delete = Some(i);
                        }
                        if ui.small_button("save").on_hover_text("overwrite with the visible channels").clicked() {
                            overwrite = Some(i);
                        }
                    });
                });
            }
            if let Some(i) = load {
                if self.active == Some(i) {
                    self.active = None;
                } else {
                    self.load(i);
                }
            }
            if let Some(i) = overwrite {
                let name = self.sets[i].name.clone();
                self.save(&name, listed);
            }
            if let Some(i) = delete {
                self.delete(i);
            }
            ui.separator();
            let n = self.visible(listed).len();
            ui.horizontal(|ui| {
                let r = ui.add(
                    egui::TextEdit::singleline(&mut self.draft)
                        .hint_text("new set name")
                        .desired_width(120.0),
                );
                let go = ui
                    .add_enabled(n > 0 && !self.draft.trim().is_empty(), egui::Button::new("save"))
                    .on_hover_text(format!("save the {n} visible channel(s) as a set"))
                    .clicked()
                    || (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
                if go {
                    let name = std::mem::take(&mut self.draft);
                    if !self.save(&name, listed) {
                        self.draft = name;
                    }
                }
            });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    fn graph() -> (EditorGraph, u64, u64) {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::Transform(Default::default()), egui::Pos2::ZERO);
        let b = g.add(NodeKind::Transform(Default::default()), egui::Pos2::ZERO);
        for (id, ch) in [(a, Channel::TranslateX), (a, Channel::Rotate), (b, Channel::ScaleX)] {
            g.nodes.get_mut(&id).unwrap().anim.entry(ch).or_default().set_key(0u64, 1.0);
        }
        (g, a, b)
    }
    fn lbl(_: u64, ch: Channel) -> String {
        crate::curves::ch_label(ch).into_owned()
    }

    #[test]
    fn selection_then_pins_then_sets() {
        let (g, a, b) = graph();
        let mut v = ChannelView::default();
        assert_eq!(v.listed(&g, &[], lbl).len(), 3);
        assert_eq!(v.listed(&g, &[a], lbl), vec![(a, Channel::Rotate), (a, Channel::TranslateX)]);
        v.pinned.insert((b, Channel::ScaleX));
        assert_eq!(v.listed(&g, &[a], lbl).len(), 3);
    }

    #[test]
    fn a_saved_set_is_the_visible_channels_and_overrides_the_selection() {
        let (g, a, b) = graph();
        let mut v = ChannelView::default();
        let listed = v.listed(&g, &[], lbl);
        v.hidden.insert((a, Channel::Rotate));
        assert!(v.save("moves", &listed));
        assert_eq!(v.active, Some(0));
        assert_eq!(v.sets[0].chans.len(), 2);
        // Selecting only b changes nothing while the set is showing.
        assert_eq!(v.listed(&g, &[b], lbl), vec![(a, Channel::TranslateX), (b, Channel::ScaleX)]);
        v.active = None;
        assert_eq!(v.listed(&g, &[b], lbl), vec![(b, Channel::ScaleX)]);
        // Same name overwrites; empty saves nothing.
        v.hidden.clear();
        assert!(v.save("moves", &listed));
        assert_eq!((v.sets.len(), v.sets[0].chans.len()), (1, 3));
        assert!(!v.save("  ", &listed));
    }

    #[test]
    fn delete_keeps_the_active_index_pointing_at_the_same_set() {
        let (g, ..) = graph();
        let mut v = ChannelView::default();
        let listed = v.listed(&g, &[], lbl);
        v.save("one", &listed);
        v.save("two", &listed);
        assert_eq!(v.active, Some(1));
        v.delete(0);
        assert_eq!((v.active, v.sets[0].name.as_str()), (Some(0), "two"));
        v.delete(0);
        assert_eq!(v.active, None);
    }

    #[test]
    fn sets_round_trip_through_json() {
        let s = vec![ChanSet { name: "x".into(), chans: vec![(3, Channel::Rotate)] }];
        let back: Vec<ChanSet> = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
        assert_eq!(back, s);
    }
}
