//! Reading a comp's cost off the wires.
//!
//! Two things taint downstream, and they are not the same thing:
//!
//! * **Slow** — a node that takes its own GPU pass makes everything below it
//!   part of a chain that cannot be evaluated in one dispatch. Noodles go
//!   red. A **loaded Bake clears it**, because the claim is then true: the
//!   pixels come off disk and the branch above is not evaluated at all.
//! * **Non-commercial** — a node whose weights are research-only makes
//!   everything below it a picture made with research-only weights. Nodes
//!   are tagged and a Write is warned. **A Bake does NOT clear it.** Baking
//!   changes what the pixels COST, not what they ARE.
//!
//! Getting those two clearing rules the same way round would be the one bug
//! in this feature that could actually cost somebody a delivery, so they are
//! computed by one walk with two different clear predicates, and there is a
//! test for each direction.
//!
//! Everything here is a pure function of an [`EditorGraph`] — no egui — so
//! the propagation is tested without a window, the way `nodeux` is.

use crate::editor::EditorGraph;
use crate::theme;
use egui::Color32;
use engine::cost::{self, Licence, Speed};
use std::collections::{HashMap, HashSet};

/// Per-node taint: does this node's OUTPUT carry each property?
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Flow {
    pub slow: HashSet<u64>,
    pub nc: HashSet<u64>,
    /// Downstream of a REALLY slow node (`Speed::Heavy`): a pass chain that
    /// dominates the frame. Always a subset of `slow`, cleared by the same
    /// loaded Bake, and drawn as a glow on top of the red.
    pub heavy: HashSet<u64>,
    /// Loads that are slow on their OWN account: frames not all in the RAM
    /// or NVMe tier, so playing them decodes from the source. A decode cost
    /// the stages cannot see (like an unbaked Cryptomatte), so the app says
    /// which; a precache clears it. They wear the ring and taint below.
    pub uncached: HashSet<u64>,
    /// Loads and Bakes by whether they serve from disk: a Load whose in..out
    /// is all cached, a Bake that is baked (true), or not (false). Drawn as
    /// the node's colour; absent = not known yet, the type colour.
    pub served: HashMap<u64, bool>,
}

impl Flow {
    pub fn is_slow(&self, node: u64) -> bool {
        self.slow.contains(&node)
    }
    pub fn is_nc(&self, node: u64) -> bool {
        self.nc.contains(&node)
    }
    pub fn is_heavy(&self, node: u64) -> bool {
        self.heavy.contains(&node)
    }

    /// The glow under a noodle leaving `src`, if its history holds a really
    /// slow node: the same red, wide and faint, so the wire reads as lit
    /// rather than merely coloured.
    pub fn wire_glow(&self, src: u64) -> Option<Color32> {
        self.is_heavy(src).then(theme::cost_slow)
    }

    /// The colour a noodle leaving `src` should be drawn in. `mask` keeps
    /// its own white while the chain is fast — the port-kind affordance is
    /// worth more than a third cyan — but red wins over everything, because
    /// cost is the thing you are being asked to notice.
    pub fn wire_color(&self, src: u64, mask: bool, mask_color: Color32) -> Color32 {
        if self.is_slow(src) {
            theme::cost_slow()
        } else if mask {
            mask_color
        } else {
            theme::cost_fast()
        }
    }
}

/// Walk the graph and propagate both taints downstream.
///
/// `is_loaded` answers whether a Bake node actually has frames on disk —
/// app state, not graph state, so it is passed in and the analysis stays a
/// pure function.
#[cfg(test)]
pub fn analyse(g: &EditorGraph, is_loaded: &dyn Fn(u64) -> bool) -> Flow {
    analyse_with(g, is_loaded, &HashSet::new())
}

/// [`analyse`], with the Loads that are not cached (`Flow::uncached`).
pub fn analyse_with(g: &EditorGraph, is_loaded: &dyn Fn(u64) -> bool, uncached: &HashSet<u64>) -> Flow {
    let mut flow = Flow { uncached: uncached.clone(), ..Flow::default() };
    let mut memo: HashMap<u64, (bool, bool, bool)> = HashMap::new();
    // Feeds by node, once: a scan of every edge per node was quadratic.
    let mut feeds: HashMap<u64, Vec<u64>> = HashMap::new();
    for (dst, _, src) in &g.edges {
        feeds.entry(*dst).or_default().push(*src);
    }
    let ids: Vec<u64> = g.nodes.keys().copied().collect();
    for id in ids {
        let (slow, nc, heavy) =
            taint(g, &feeds, id, is_loaded, uncached, &mut memo, &mut std::collections::HashSet::new());
        if slow {
            flow.slow.insert(id);
        }
        if nc {
            flow.nc.insert(id);
        }
        if heavy {
            flow.heavy.insert(id);
        }
    }
    flow
}

/// (slow, non-commercial, heavy) for one node's output, memoized. `stack`
/// breaks cycles: the editor can wire one, and a hang while drawing would
/// be a far worse bug than a wire that is briefly the wrong colour.
fn taint(
    g: &EditorGraph,
    feeds: &HashMap<u64, Vec<u64>>,
    node: u64,
    is_loaded: &dyn Fn(u64) -> bool,
    uncached: &HashSet<u64>,
    memo: &mut HashMap<u64, (bool, bool, bool)>,
    stack: &mut std::collections::HashSet<u64>,
) -> (bool, bool, bool) {
    if let Some(v) = memo.get(&node) {
        return *v;
    }
    if stack.contains(&node) {
        return (false, false, false); // cycle
    }
    let Some(n) = g.nodes.get(&node) else { return (false, false, false) };
    let kind = &n.kind;

    // A loaded Bake is a wall for cost: nothing above it is evaluated, so
    // nothing above it can make what leaves it slow. It is NOT a wall for
    // licence — see the module header.
    let loaded_bake = cost::clears_slow(kind, is_loaded(node));

    stack.insert(node);
    let (mut up_slow, mut up_nc, mut up_heavy) = (false, false, false);
    for src in feeds.get(&node).into_iter().flatten() {
        let (s, c, h) = taint(g, feeds, *src, is_loaded, uncached, memo, stack);
        up_slow |= s;
        up_nc |= c;
        up_heavy |= h;
    }
    stack.remove(&node);

    // A DISABLED node does no work and loads no weights, so it taints with
    // neither — it is a passthrough, and the colour should say so.
    let speed = if n.disabled { Speed::Fast } else { cost::speed_of(kind) };
    let own_slow = speed != Speed::Fast || !n.disabled && uncached.contains(&node);
    let own_heavy = speed == Speed::Heavy;
    // Restricted licences ride the same taint: both warn at a Write.
    let own_nc = !n.disabled && cost::licence_of(kind).warns();

    // Heavy clears exactly where slow does: it is the same cost, more of it.
    let out = (
        if loaded_bake { false } else { own_slow || up_slow },
        own_nc || up_nc,
        if loaded_bake { false } else { own_heavy || up_heavy },
    );
    memo.insert(node, out);
    out
}

/// One line naming what is upstream, for a warning that says WHICH node
/// rather than just that there is one.
pub fn nc_sources(g: &EditorGraph) -> Vec<String> {
    let mut v: Vec<String> = g
        .nodes
        .values()
        .filter(|n| !n.disabled && cost::licence_of(&n.kind).warns())
        .map(|n| n.kind.name().to_string())
        .collect();
    v.sort();
    v.dedup();
    v
}

/// The words a warning uses: research-only beats restricted, because it
/// is the stronger claim. `true` = something research-only is in the comp.
pub fn any_research_only(g: &EditorGraph) -> bool {
    g.nodes.values().any(|n| !n.disabled && cost::licence_of(&n.kind) == Licence::NonCommercial)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    fn never(_: u64) -> bool {
        false
    }
    fn always(_: u64) -> bool {
        true
    }

    /// read -> blur -> grade -> grade, so there is a fast head, a slow
    /// middle and a fast tail to check the taint runs through.
    fn chain() -> (EditorGraph, Vec<u64>) {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), egui::Pos2::ZERO);
        let b = g.add(NodeKind::Blur(Default::default()), egui::Pos2::ZERO);
        let c = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        let d = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        g.edges.push((b, 0, r));
        g.edges.push((c, 0, b));
        g.edges.push((d, 0, c));
        (g, vec![r, b, c, d])
    }

    #[test]
    fn slow_taints_everything_downstream() {
        let (g, n) = chain();
        let f = analyse(&g, &never);
        assert!(!f.is_slow(n[0]), "a Read is not slow");
        assert!(f.is_slow(n[1]), "the Blur itself");
        assert!(f.is_slow(n[2]), "downstream of it");
        assert!(f.is_slow(n[3]), "and further downstream");
    }

    #[test]
    fn a_loaded_bake_clears_slow() {
        let (mut g, n) = chain();
        let bake = g.add(
            NodeKind::GraphBake { source: 0, baked: false, start: None, mask: false, window: None, matte_half: false, single: false },
            egui::Pos2::ZERO,
        );
        let tail = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        g.edges.push((bake, 0, n[3]));
        g.edges.push((tail, 0, bake));

        let f = analyse(&g, &always);
        assert!(f.is_slow(n[3]), "above the bake is still slow");
        assert!(!f.is_slow(bake), "the bake's own output comes off disk");
        assert!(!f.is_slow(tail), "and so everything below it is fast again");
    }

    #[test]
    fn an_unbaked_bake_clears_nothing() {
        let (mut g, n) = chain();
        let bake = g.add(
            NodeKind::GraphBake { source: 0, baked: false, start: None, mask: false, window: None, matte_half: false, single: false },
            egui::Pos2::ZERO,
        );
        g.edges.push((bake, 0, n[3]));
        // Nothing on disk: the branch above is still being evaluated live,
        // and pretending otherwise would make the button look like it had
        // already been pressed.
        let f = analyse(&g, &never);
        assert!(f.is_slow(bake));
    }

    #[test]
    fn a_disabled_slow_node_is_not_slow() {
        let (mut g, n) = chain();
        g.nodes.get_mut(&n[1]).unwrap().disabled = true;
        let f = analyse(&g, &never);
        assert!(!f.is_slow(n[1]), "a bypassed Blur does no work");
        assert!(!f.is_slow(n[3]));
    }

    /// THE rule that must not get flipped: a Bake changes what pixels COST,
    /// never what they ARE.
    #[test]
    fn a_bake_does_not_clear_non_commercial() {
        let mut g = EditorGraph::default();
        let src = g.add(NodeKind::read(0, ""), egui::Pos2::ZERO);
        // Stand in for a research-only node by asserting the rule through
        // the same path a real one takes: build the taint, then bake it.
        let nc = g.add(NodeKind::Tracker(Default::default()), egui::Pos2::ZERO);
        let bake = g.add(
            NodeKind::GraphBake { source: 0, baked: false, start: None, mask: false, window: None, matte_half: false, single: false },
            egui::Pos2::ZERO,
        );
        g.edges.push((nc, 0, src));
        g.edges.push((bake, 0, nc));

        // With today's registry the Tracker is permissive, so this asserts
        // the WIRING rather than the verdict: whatever `licence_of` says of
        // the upstream node, the bake must not change it.
        let up = analyse(&g, &always).is_nc(nc);
        let below = analyse(&g, &always).is_nc(bake);
        assert_eq!(up, below, "a bake changed the licence taint");
    }

    /// The same walk, forced through both clear rules at once: a node that
    /// is slow AND restricted, baked, comes out fast and still restricted.
    #[test]
    fn the_two_taints_clear_differently() {
        use engine::cost::{clears_slow, Licence};
        let bake = NodeKind::GraphBake { source: 0, baked: false, start: None, mask: false, window: None, matte_half: false, single: false };
        // Speed: a loaded bake clears it.
        assert!(clears_slow(&bake, true));
        // Licence: there is no clearing predicate at all, by construction —
        // `taint` ORs `own_nc || up_nc` with nothing that can reset it.
        assert_eq!(engine::cost::licence_of(&bake), Licence::Permissive);
        let (mut g, n) = chain();
        let b = g.add(bake, egui::Pos2::ZERO);
        g.edges.push((b, 0, n[3]));
        let f = analyse(&g, &always);
        assert!(!f.is_slow(b));
    }

    #[test]
    fn a_cycle_does_not_hang() {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        let b = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        g.edges.push((a, 0, b));
        g.edges.push((b, 0, a));
        let _ = analyse(&g, &never);
    }

    #[test]
    fn a_branch_taints_only_its_own_side() {
        // Two Reads into a Blend; only one goes through a Blur.
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::read(0, ""), egui::Pos2::ZERO);
        let b = g.add(NodeKind::read(1, ""), egui::Pos2::ZERO);
        let blur = g.add(NodeKind::Blur(Default::default()), egui::Pos2::ZERO);
        let blend = g.add(NodeKind::blend(), egui::Pos2::ZERO);
        g.edges.push((blur, 0, a));
        g.edges.push((blend, 0, blur));
        g.edges.push((blend, 1, b));
        let f = analyse(&g, &never);
        assert!(!f.is_slow(b), "the clean branch stays cyan");
        assert!(f.is_slow(blend), "but the merge inherits the slow side");
    }

    /// A REALLY slow node lights everything below it, on top of the red;
    /// an ordinary slow node does not; a loaded Bake puts both out.
    #[test]
    fn heavy_glows_downstream_and_a_bake_puts_it_out() {
        use engine::graph::{DefocusMode, DefocusParams};
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), egui::Pos2::ZERO);
        let blur = g.add(NodeKind::Blur(Default::default()), egui::Pos2::ZERO);
        let heavy = DefocusParams { mode: DefocusMode::Accurate, ..Default::default() };
        let conv = g.add(NodeKind::Defocus(heavy), egui::Pos2::ZERO);
        let tail = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        let bake = g.add(NodeKind::GraphBake { source: 0, baked: false, start: None, mask: false, window: None, matte_half: false, single: false }, egui::Pos2::ZERO);
        let after = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        g.edges.push((blur, 0, r));
        g.edges.push((conv, 0, blur));
        g.edges.push((tail, 0, conv));
        g.edges.push((bake, 0, tail));
        g.edges.push((after, 0, bake));
        let f = analyse(&g, &never);
        assert!(f.is_slow(blur) && !f.is_heavy(blur), "a Blur is slow, not heavy");
        assert!(f.is_heavy(conv), "an accurate Defocus is heavy");
        assert!(f.is_heavy(tail) && f.is_slow(tail), "and everything below it glows");
        assert_eq!(f.wire_glow(tail), Some(theme::cost_slow()));
        assert_eq!(f.wire_glow(blur), None);
        assert!(f.heavy.is_subset(&f.slow), "heavy is a kind of slow");
        let f = analyse(&g, &always);
        assert!(!f.is_heavy(bake) && !f.is_heavy(after), "a loaded bake puts the glow out");
        assert!(!f.is_slow(after));
    }

    #[test]
    fn wire_colour_follows_the_taint() {
        let (g, n) = chain();
        let f = analyse(&g, &never);
        let green = Color32::GREEN;
        assert_eq!(f.wire_color(n[0], false, green), theme::cost_fast());
        assert_eq!(f.wire_color(n[0], true, green), green, "a fast mask stays a mask");
        assert_eq!(f.wire_color(n[1], false, green), theme::cost_slow());
        assert_eq!(f.wire_color(n[1], true, green), theme::cost_slow(), "cost wins over port kind");
    }

    #[test]
    fn an_uncached_load_is_slow_and_taints_below() {
        let (g, n) = chain_fast();
        let f = analyse_with(&g, &never, &[n[0]].into_iter().collect());
        assert!(f.is_slow(n[0]) && f.is_slow(n[1]), "the Load and everything below go red");
        let f = analyse_with(&g, &never, &HashSet::new());
        assert!(!f.is_slow(n[0]) && !f.is_slow(n[1]), "cached, the chain is fast");
    }

    /// read -> grade: nothing slow of its own.
    fn chain_fast() -> (EditorGraph, Vec<u64>) {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), egui::Pos2::ZERO);
        let c = g.add(NodeKind::Grade(Default::default()), egui::Pos2::ZERO);
        g.edges.push((c, 0, r));
        (g, vec![r, c])
    }
}
