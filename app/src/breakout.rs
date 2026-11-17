//! The Breakout / Recombine pair — the AOV comb as a HARP.
//!
//! `aovs::build_rig` (the select + Add ladder) stays for people who want
//! the classic shape; this is the dedicated instrument. A [`NodeKind::Breakout`]
//! is one wide node on the fat cable with an output nodule per chosen
//! layer, evenly pitched by its `spacing`; its paired
//! [`NodeKind::Recombine`] sums the strings back together and rides
//! X-LOCKED below it, so the noodles between them hang straight like harp
//! strings — and a Grade spliced into a string is a graded AOV.
//!
//! Nothing downstream of the editor knows any of this:
//!
//! - Behind each output nodule sits a hidden, ordinary **ChannelSelect
//!   kid** ([`sync`] manages them; the editor never draws them). Slots,
//!   decode, caching, instancing: all the select machinery, untouched.
//! - [`expand`] re-points every consumer of a Breakout output at its kid
//!   and drops the Breakout, before `to_engine_graph_at_f` builds the
//!   engine graph — the lumabands/tools/Dot trick, one more time.
//! - The Recombine is a real fused node (`codegen` sums its inputs), so
//!   the rebuilt beauty costs one expression, not a ladder of Blends.

use crate::editor::EditorGraph;
use engine::graph::{BreakoutParams, NodeKind, RecombineParams};
use egui::Pos2;

/// Vertical gap the receiver spawns at, graph units.
const DROP: f32 = 300.0;

/// The porch a bar grows past its strings: the Breakout's stretches LEFT
/// for its cable input, the receiver's RIGHT for its output — so the
/// through-ports read as bookends, not as one more string.
pub const END_EXT: f32 = 34.0;

/// Pitch between a merge bar's strings, graph units at size 1. Fixed: a
/// bar's layout scales with its SIZE (the selection handles), where a
/// harp's pitch is its own param.
pub const BAR_PITCH: f32 = 40.0;

/// A merge drawn as a harp bar — Extract, Layer, a GenAI Pipeline: one wide slab with its
/// inputs pitched across the top like strings, its outputs hanging below,
/// and at most one bookend input on the left porch. Only for kinds whose
/// input count GROWS: a Blend is two inputs and a matte for good, and
/// reads better as an ordinary node (per user).
struct Bar {
    /// Input indices along the top, left to right.
    top: Vec<usize>,
    /// Outputs along the bottom.
    bottom: usize,
    /// The input that lands on the left porch, if any.
    porch: Option<usize>,
}

fn bar(kind: &NodeKind) -> Option<Bar> {
    Some(match kind {
        // The render enters on the porch; each matte stands over the
        // output it isolates.
        NodeKind::Extract(p) => {
            Bar { top: (1..=p.pairs()).collect(), bottom: p.pairs(), porch: Some(0) }
        }
        // Stack order, top layer leftmost.
        NodeKind::Layer(_) => {
            Bar { top: (0..kind.input_count()).collect(), bottom: 1, porch: None }
        }
        // A GenAI Pipeline: its pictures on the left, its mattes on the
        // right, each side in the order they were added (an input's index
        // never changes when another is added). A one-model node keeps the
        // ordinary disc: its inputs are its model's, fixed.
        NodeKind::GenAi(p) if p.model.is_none() => {
            let n = kind.input_count();
            let side = |k: genai::InKind| (0..n).filter(move |&i| p.pipeline.in_kind(i) == k);
            let top = side(genai::InKind::Image).chain(side(genai::InKind::Matte)).collect();
            Bar { top, bottom: 1, porch: None }
        }
        _ => return None,
    })
}

/// Is this kind drawn as a merge bar (not a harp half)?
pub fn is_bar(kind: &NodeKind) -> bool {
    bar(kind).is_some()
}

/// x of string `k` of `n`, centred on the node.
fn pitched(pos: Pos2, k: usize, n: usize, size: f32) -> f32 {
    let k = k.min(n.saturating_sub(1)) as f32;
    pos.x + (k - (n.max(1) as f32 - 1.0) / 2.0) * BAR_PITCH * size
}

/// Where a merge bar's input rests along its top edge (a string, or the
/// left porch), x in graph units. `None` for every other kind.
pub fn bar_in_x(kind: &NodeKind, pos: Pos2, input: usize, size: f32) -> Option<f32> {
    let b = bar(kind)?;
    let n = b.top.len().max(b.bottom).max(1);
    if b.porch == Some(input) {
        return Some(pitched(pos, 0, n, size) - END_EXT * size);
    }
    // The strings share the widest row's span, so an Extract's mattes sit
    // exactly over their outputs.
    let k = b.top.iter().position(|i| *i == input)?;
    let offset = (n - b.top.len().max(1)) / 2;
    Some(pitched(pos, k + offset, n, size))
}

/// The x of every string of a merge bar, in input order (graph units).
pub fn bar_strings_x(kind: &NodeKind, pos: Pos2, size: f32) -> Vec<(usize, f32)> {
    let Some(b) = bar(kind) else { return Vec::new() };
    b.top.iter().filter_map(|i| bar_in_x(kind, pos, *i, size).map(|x| (*i, x))).collect()
}

/// Where a merge bar's output `slot` hangs, x in graph units.
pub fn bar_out_x(kind: &NodeKind, pos: Pos2, slot: usize, size: f32) -> Option<f32> {
    let b = bar(kind)?;
    Some(pitched(pos, slot, b.bottom.max(1), size))
}

/// A wide node's body: (half extents, centre x-offset) in graph units at
/// size 1. The offset is the porch: the drawn slab is NOT centred on the
/// node — it leans the way its end port lives. Shared by the body draw,
/// the body hit test, `node_rects` and the selection bbox, so what you
/// see is exactly what you can grab.
pub fn wide_rect(kind: &NodeKind) -> Option<(egui::Vec2, f32)> {
    if let Some(b) = bar(kind) {
        let n = b.top.len().max(b.bottom).max(1);
        let span_half = (n - 1) as f32 * 0.5 * BAR_PITCH;
        let porch = if b.porch.is_some() { END_EXT } else { 0.0 };
        return Some((egui::Vec2::new(span_half + 20.0 + porch * 0.5, 13.0), -porch * 0.5));
    }
    // (left porch, right porch): the Breakout takes the fat cable on its
    // left and hangs its strings; the receiver takes the base rgba on its
    // left AND leaves with the rebuilt beauty on its right, so it wears
    // both bookends.
    let (n, spacing, porches, offs) = match kind {
        NodeKind::Breakout(p) => (p.strings(), p.spacing, (true, false), &p.offsets),
        NodeKind::Recombine(p) => (p.strings(), p.spacing, (true, true), &p.offsets),
        _ => return None,
    };
    // The slab wraps WHERE ITS STRINGS ARE, not how far they have been
    // nudged: a string moved about inside the comb leaves the ends where
    // they were, so the bar holds still until something actually passes
    // the end of it. (Sizing off the nudges instead made the bar stretch
    // with every drag and snap back on each reorder — a ratchet.)
    let spacing = spacing.max(20.0);
    let at = |i: usize| -> f32 {
        (i as f32 - (n as f32 - 1.0) / 2.0) * spacing
            + offs.get(i).copied().unwrap_or(0.0)
    };
    let (lo, hi) = (0..n).fold((f32::MAX, f32::MIN), |(lo, hi), i| {
        (lo.min(at(i)), hi.max(at(i)))
    });
    let (lo, hi) = if n == 0 { (0.0, 0.0) } else { (lo, hi) };
    let (l, r) = (
        if porches.0 { END_EXT } else { 0.0 },
        if porches.1 { END_EXT } else { 0.0 },
    );
    let half = egui::Vec2::new((hi - lo) * 0.5 + 20.0 + (l + r) * 0.5, 13.0);
    Some((half, (lo + hi) * 0.5 + (r - l) * 0.5))
}

/// Just the extents, for callers that already anchor on the node centre.
pub fn wide_half(kind: &NodeKind) -> Option<egui::Vec2> {
    wide_rect(kind).map(|(h, _)| h)
}

/// The bar-end port's x in graph units: the Breakout's cable input on the
/// left porch, the receiver's output on the right one.
pub fn end_x(kind: &NodeKind, pos: Pos2, size: f32) -> Option<f32> {
    match kind {
        NodeKind::Breakout(_) => {
            string_x(kind, pos, 0, size).map(|x| x - END_EXT * size)
        }
        NodeKind::Recombine(p) => {
            let last = p.strings() - 1;
            string_x(kind, pos, last, size).map(|x| x + END_EXT * size)
        }
        _ => None,
    }
}

/// Where the receiver's BASE input rests: the left bookend, mirroring the
/// Breakout's cable input above it. `None` for every other kind.
pub fn base_x(kind: &NodeKind, pos: Pos2, size: f32) -> Option<f32> {
    match kind {
        NodeKind::Recombine(_) => {
            string_x(kind, pos, 0, size).map(|x| x - END_EXT * size)
        }
        _ => None,
    }
}

/// Where output `slot` (Breakout) or input `slot` (Recombine) rests, in
/// graph units — the strings' pitch. `None` for every other kind.
pub fn string_x(kind: &NodeKind, pos: Pos2, slot: usize, size: f32) -> Option<f32> {
    let (n, spacing, off) = match kind {
        NodeKind::Breakout(p) => {
            (p.strings(), p.spacing, p.offsets.get(slot).copied().unwrap_or(0.0))
        }
        NodeKind::Recombine(p) => (
            p.strings(),
            p.spacing,
            p.offsets.get(slot).copied().unwrap_or(0.0),
        ),
        _ => return None,
    };
    let k = slot.min(n.saturating_sub(1)) as f32;
    Some(pos.x + ((k - (n as f32 - 1.0) / 2.0) * spacing.max(20.0) + off) * size)
}

/// The other bar of any harp in `ids`: selecting either half of a pair
/// puts BOTH under the transform bbox, so the bars scale as one
/// instrument. (Nodes spliced into the strings stay individual — they are
/// not companions, just neighbours.)
pub fn companions(graph: &EditorGraph, ids: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    for id in ids {
        let pair = match graph.node(*id).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => p.pair,
            Some(NodeKind::Recombine(p)) => p.pair,
            _ => 0,
        };
        if pair != 0
            && graph.nodes.contains_key(&pair)
            && !ids.contains(&pair)
            && !out.contains(&pair)
        {
            out.push(pair);
        }
    }
    out
}

/// Every hidden ChannelSelect kid in the graph — what the editor's hit
/// tests, marquee, magnetism and wire drawing all leave alone.
pub fn kid_set(graph: &EditorGraph) -> std::collections::HashSet<u64> {
    graph
        .nodes
        .values()
        .filter_map(|n| match &n.kind {
            NodeKind::Breakout(p) => {
                Some(p.kids.iter().chain(p.remainder_kids.iter()).copied())
            }
            _ => None,
        })
        .flatten()
        .collect()
}

/// Keep every Breakout's hidden machinery true to its face, once per
/// frame: one ChannelSelect kid per layer (spawned, retired, re-layered,
/// parked on the node, fed by the node's own cable), the paired Recombine
/// x-locked underneath with its input count and pitch mirrored, and
/// freshly grown strings wired down to it. Deliberately idempotent — a
/// frame where nothing changed writes nothing.
pub fn sync(graph: &mut EditorGraph) {
    let breakouts: Vec<u64> = graph
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n.kind, NodeKind::Breakout(_)))
        .map(|(id, _)| *id)
        .collect();
    for b in breakouts {
        let (layers, mut kids, spacing, pair, pos, remainder, mut rkids, strings, boffsets) = {
            let Some(n) = graph.node(b) else { continue };
            let NodeKind::Breakout(p) = &n.kind else { continue };
            (
                p.layers.clone(),
                p.kids.clone(),
                p.spacing,
                p.pair,
                n.pos,
                p.remainder,
                p.remainder_kids.clone(),
                p.strings(),
                p.offsets.clone(),
            )
        };
        let feed = graph.input_of(b, 0);
        // The kid roster follows the layer list.
        kids.retain(|k| {
            graph.nodes.get(k).is_some_and(|n| matches!(n.kind, NodeKind::ChannelSelect(_)))
        });
        while kids.len() > layers.len() {
            if let Some(k) = kids.pop() {
                graph.remove(k);
            }
        }
        while kids.len() < layers.len() {
            kids.push(graph.add(NodeKind::ChannelSelect(Default::default()), pos));
        }
        for (i, k) in kids.iter().enumerate() {
            if let Some(n) = graph.node_mut(*k) {
                if n.pos != pos {
                    n.pos = pos;
                }
                if let NodeKind::ChannelSelect(p) = &mut n.kind {
                    if p.layer != layers[i] {
                        p.layer = layers[i].clone();
                    }
                }
            }
            match feed {
                Some(src) if graph.input_of(*k, 0) != Some(src) => {
                    graph.connect(*k, 0, src);
                }
                None if graph.input_of(*k, 0).is_some() => {
                    graph.disconnect_input(*k, 0);
                }
                _ => {}
            }
        }
        // The BASE RGBA kid: one hidden ChannelSelect on the node's own
        // cable, reading the file's plain rgba. Two things want it — the
        // remainder (below) and a receiver taking its ALPHA off the beauty
        // — so it exists when either does, and `remainder_kids` holds it
        // first with the remainder's signed sum behind it.
        let alpha_base = pair != 0
            && matches!(
                graph.node(pair).map(|n| &n.kind),
                Some(NodeKind::Recombine(rp)) if rp.alpha == engine::graph::AlphaFrom::Base
            );
        let needs_base = remainder || alpha_base;
        rkids.retain(|k| graph.nodes.contains_key(k));
        if needs_base {
            let base_ok = rkids
                .first()
                .and_then(|k| graph.node(*k))
                .is_some_and(|n| matches!(n.kind, NodeKind::ChannelSelect(_)));
            if !base_ok {
                for k in rkids.drain(..) {
                    graph.remove(k);
                }
                rkids.push(graph.add(NodeKind::ChannelSelect(Default::default()), pos));
            }
            let base = rkids[0];
            if let Some(n) = graph.node_mut(base) {
                if n.pos != pos {
                    n.pos = pos;
                }
                if let NodeKind::ChannelSelect(p) = &mut n.kind {
                    if !p.layer.is_empty() {
                        p.layer = String::new(); // the base rgba
                    }
                }
            }
            match feed {
                Some(src) if graph.input_of(base, 0) != Some(src) => {
                    graph.connect(base, 0, src);
                }
                None if graph.input_of(base, 0).is_some() => {
                    graph.disconnect_input(base, 0);
                }
                _ => {}
            }
        } else {
            for k in rkids.drain(..) {
                graph.remove(k);
            }
        }
        // The remainder's own machinery: a SIGNED Recombine (base plus,
        // every layer kid minus) — what is left when the chosen AOVs are
        // taken back out of the beauty. Capped by the fused sum's 16
        // inputs, so it wants at most 15 layers.
        if remainder {
            let sum_ok = rkids
                .get(1)
                .and_then(|k| graph.node(*k))
                .is_some_and(|n| matches!(n.kind, NodeKind::Recombine(_)));
            if !sum_ok {
                for k in rkids.split_off(rkids.len().min(1)) {
                    graph.remove(k);
                }
                rkids.push(graph.add(
                    NodeKind::Recombine(RecombineParams {
                        n: 1,
                        // It is arithmetic, not a rebuild: its alpha is
                        // whatever base-minus-passes comes to.
                        alpha: engine::graph::AlphaFrom::Sum,
                        ..Default::default()
                    }),
                    pos,
                ));
            }
            let (base, rsum) = (rkids[0], rkids[1]);
            if let Some(n) = graph.node_mut(rsum) {
                if n.pos != pos {
                    n.pos = pos;
                }
            }
            let want_n = (layers.len() + 1).min(16) as u32;
            let want_sub: Vec<bool> =
                std::iter::once(false).chain(layers.iter().map(|_| true)).collect();
            if let Some(n) = graph.node_mut(rsum) {
                if let NodeKind::Recombine(p) = &mut n.kind {
                    if p.n != want_n {
                        p.n = want_n;
                    }
                    if p.sub != want_sub {
                        p.sub = want_sub;
                    }
                }
            }
            if graph.input_of(rsum, 0) != Some(base) {
                graph.connect(rsum, 0, base);
            }
            for (i, k) in kids.iter().enumerate() {
                if i + 1 < 16 && graph.input_of(rsum, i + 1) != Some(*k) {
                    graph.connect(rsum, i + 1, *k);
                }
            }
        } else {
            for k in rkids.split_off(rkids.len().min(1)) {
                graph.remove(k);
            }
        }
        if let Some(NodeKind::Breakout(p)) = graph.node_mut(b).map(|n| &mut n.kind) {
            if p.kids != kids {
                p.kids = kids;
            }
            if p.remainder_kids != rkids {
                p.remainder_kids = rkids;
            }
        }
        // The pair: x-locked, pitch mirrored, one input per string; inputs
        // that exist because a layer just did are wired down — inputs the
        // hand cut stay cut. It also FOLLOWS the breakout's region, so a
        // region wash always wraps the whole instrument.
        if pair != 0 {
            let (br, pr) = (
                crate::regions::region_of(graph, b),
                crate::regions::region_of(graph, pair),
            );
            match br {
                Some(rid) if pr != Some(rid) => {
                    crate::regions::adopt(graph, rid, &[pair]);
                }
                None => {
                    if let Some(prid) = pr {
                        if let Some(r) =
                            graph.regions.iter_mut().find(|r| r.id == prid)
                        {
                            r.members.retain(|m| *m != pair);
                        }
                        crate::regions::prune(graph);
                    }
                }
                _ => {}
            }
        }
        if pair != 0 {
            let n_new = strings as u32;
            let old_n = match graph.node(pair).map(|n| &n.kind) {
                Some(NodeKind::Recombine(p)) => p.n,
                _ => continue,
            };
            let bsize = graph.node(b).map(|n| n.size).unwrap_or(1.0);
            if let Some(rn) = graph.node_mut(pair) {
                rn.pos.x = pos.x;
                if rn.size != bsize {
                    rn.size = bsize;
                }
                if let NodeKind::Recombine(p) = &mut rn.kind {
                    if p.n != n_new {
                        p.n = n_new;
                    }
                    if p.spacing != spacing {
                        p.spacing = spacing;
                    }
                    // Nudges mirror too: the strings hang straight.
                    let boffs = boffsets.clone();
                    if p.offsets != boffs {
                        p.offsets = boffs;
                    }
                }
            }
            for k in 0..n_new as usize {
                if k >= old_n as usize
                    && k < strings
                    && graph.input_of(pair, k).is_none()
                {
                    graph.connect_out(pair, k, b, k as u32);
                }
            }
        }
    }
}

/// The x-lock alone, cheap enough to run twice a frame: once in [`sync`]
/// and once after the editor's gestures apply, so a dragged receiver never
/// DRAWS off its column.
pub fn lock_x(graph: &mut EditorGraph) {
    let pairs: Vec<(u64, f32)> = graph
        .nodes
        .values()
        .filter_map(|n| match &n.kind {
            NodeKind::Breakout(p) if p.pair != 0 => Some((p.pair, n.pos.x)),
            _ => None,
        })
        .collect();
    for (r, x) in pairs {
        if let Some(n) = graph.node_mut(r) {
            if n.pos.x != x {
                n.pos.x = x;
            }
        }
    }
}

/// Spawn the pair wired as a harp: Breakout at `at` over `layers`, its
/// Recombine below, every string strung. Returns (breakout, recombine).
pub fn spawn_pair(graph: &mut EditorGraph, at: Pos2, layers: Vec<String>) -> (u64, u64) {
    let n = layers.len().max(1) as u32;
    let b = graph.add(
        NodeKind::Breakout(BreakoutParams { layers, ..Default::default() }),
        at,
    );
    let r = graph.add(
        NodeKind::Recombine(RecombineParams { n, pair: b, ..Default::default() }),
        at + egui::Vec2::new(0.0, DROP),
    );
    if let Some(NodeKind::Breakout(p)) = graph.node_mut(b).map(|k| &mut k.kind) {
        p.pair = r;
    }
    for k in 0..n as usize {
        graph.connect_out(r, k, b, k as u32);
    }
    sync(graph);
    (b, r)
}

/// Move string `from` to index `to`, carrying everything that belongs to
/// it: its layer, its hidden kid, its nudge, the receiver's gain (keyed
/// curve included), the receiver input its chain lands in, and every wire
/// that reads that output.
///
/// The bar's ORDER is what it draws, so this is what a chip dragged past
/// its neighbour does. The alternative — letting the nudge keep growing —
/// leaves two strings standing on each other and stretches the slab to
/// cover a gap nothing is in.
pub fn reorder(graph: &mut EditorGraph, bar: u64, from: usize, to: usize) -> bool {
    let (mut layers, mut kids, mut offs, pair) = match graph.node(bar).map(|n| &n.kind) {
        Some(NodeKind::Breakout(p)) => {
            (p.layers.clone(), p.kids.clone(), p.offsets.clone(), p.pair)
        }
        _ => return false,
    };
    let n = layers.len();
    if from >= n || to >= n || from == to {
        return false;
    }
    // order[new] = old, and its inverse.
    let mut order: Vec<usize> = (0..n).collect();
    let moved = order.remove(from);
    order.insert(to, moved);
    let mut pos_of = vec![0usize; n];
    for (new, old) in order.iter().enumerate() {
        pos_of[*old] = new;
    }
    let take = |v: &mut Vec<f32>| -> Vec<f32> {
        v.resize(n.max(v.len()), 0.0);
        order.iter().map(|o| v[*o]).collect::<Vec<f32>>()
    };
    layers = order.iter().map(|o| layers[*o].clone()).collect();
    if kids.len() >= n {
        kids = order.iter().map(|o| kids[*o]).collect();
    }
    let new_offs = take(&mut offs);
    if let Some(NodeKind::Breakout(p)) = graph.node_mut(bar).map(|n| &mut n.kind) {
        p.layers = layers;
        if kids.len() >= n {
            p.kids = kids;
        }
        p.offsets = new_offs.clone();
    }
    // Every wire that reads a string: its output index moves with it.
    let from_bar: Vec<(u64, usize, u32)> = graph
        .edges
        .iter()
        .filter(|(_, _, s)| *s == bar)
        .map(|(d, i, _)| (*d, *i, graph.out_of(*d, *i)))
        .collect();
    for (d, i, out) in &from_bar {
        let k = *out as usize;
        if k < n && pos_of[k] != k {
            graph.connect_out(*d, *i, bar, pos_of[k] as u32);
        }
    }
    if pair == 0 {
        return true;
    }
    // The receiver: the string's INPUT (its chain lands there), its gain
    // and its keyed gain curve all move with it.
    let inputs: Vec<Option<(u64, u32)>> = (0..n)
        .map(|i| graph.input_of(pair, i).map(|src| (src, graph.out_of(pair, i))))
        .collect();
    for (new, old) in order.iter().enumerate() {
        match inputs[*old] {
            Some((src, out)) => graph.connect_out(pair, new, src, out),
            None => graph.disconnect_input(pair, new),
        }
    }
    let curves: std::collections::BTreeMap<usize, crate::anim::AnimCurve> = graph
        .node(pair)
        .map(|rn| {
            (0..n)
                .filter_map(|i| {
                    rn.anim
                        .get(&crate::anim::Channel::RecombineGain(i as u16))
                        .map(|c| (i, c.clone()))
                })
                .collect()
        })
        .unwrap_or_default();
    if let Some(rn) = graph.node_mut(pair) {
        if let NodeKind::Recombine(p) = &mut rn.kind {
            p.offsets = new_offs;
            p.gain.resize(n.max(p.gain.len()), 1.0);
            p.gain = order
                .iter()
                .map(|o| p.gain[*o])
                .chain(p.gain[n..].iter().copied())
                .collect();
        }
        for i in 0..n {
            rn.anim.remove(&crate::anim::Channel::RecombineGain(i as u16));
        }
        for (old, curve) in curves {
            rn.anim
                .insert(crate::anim::Channel::RecombineGain(pos_of[old] as u16), curve);
        }
    }
    true
}

/// A string that has been nudged PAST a neighbour changes places with it,
/// as many times as the drag went. The dragged string stays under the
/// cursor — the pitch it just gained is taken back out of its nudge — and
/// the neighbour SHUFFLES ASIDE into the slot it vacated, chain and all.
/// So the comb keeps its spacing and the slab keeps its width, instead of
/// stretching to cover a gap nothing is standing in. Returns where the
/// string ended up.
pub fn pass_reorder(graph: &mut EditorGraph, bar: u64, string: usize) -> usize {
    let mut k = string;
    for _ in 0..64 {
        let (offs, spacing, n) = match graph.node(bar).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => {
                (p.offsets.clone(), p.spacing.max(20.0), p.layers.len())
            }
            _ => return k,
        };
        if k >= n {
            return k;
        }
        // Where each string sits along the bar, pitch and nudge together.
        let u = |i: usize| i as f32 * spacing + offs.get(i).copied().unwrap_or(0.0);
        let to = if k + 1 < n && u(k) > u(k + 1) {
            k + 1
        } else if k > 0 && u(k) < u(k - 1) {
            k - 1
        } else {
            return k;
        };
        if !reorder(graph, bar, k, to) {
            return k;
        }
        // The dragged string keeps its x: it gained an index, so a pitch
        // comes back out of its nudge. The displaced one keeps its nudge
        // and therefore MOVES — one pitch, into the slot just vacated —
        // and its spliced chain goes with it.
        let step = to as f32 - k as f32;
        nudge(graph, bar, to, -step * spacing);
        let size = graph.node(bar).map(|n| n.size).unwrap_or(1.0);
        let pair = match graph.node(bar).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => p.pair,
            _ => 0,
        };
        if pair != 0 {
            if let Some(chain) = string_chain(graph, bar, pair, k) {
                let dx = -step * spacing * size;
                for id in chain {
                    if let Some(n) = graph.node_mut(id) {
                        n.pos.x += dx;
                    }
                }
            }
        }
        k = to;
    }
    k
}

/// Add `d` to string `i`'s nudge on the bar AND its receiver, so the
/// string hangs straight.
fn nudge(graph: &mut EditorGraph, bar: u64, i: usize, d: f32) {
    let pair = match graph.node(bar).map(|n| &n.kind) {
        Some(NodeKind::Breakout(p)) => p.pair,
        _ => return,
    };
    for id in [bar, pair] {
        if let Some(n) = graph.node_mut(id) {
            let offs = match &mut n.kind {
                NodeKind::Breakout(p) => Some(&mut p.offsets),
                NodeKind::Recombine(p) => Some(&mut p.offsets),
                _ => None,
            };
            if let Some(o) = offs {
                if o.len() <= i {
                    o.resize(i + 1, 0.0);
                }
                o[i] += d;
            }
        }
    }
}

/// A Breakout added bare (menu, Tab search) arrives as a harp: give it its
/// receiver. Layers start empty — the panel's tick list fills them once a
/// fat cable arrives.
pub fn after_add(graph: &mut EditorGraph, id: u64) {
    let bare = matches!(
        graph.node(id).map(|n| &n.kind),
        Some(NodeKind::Breakout(p)) if p.pair == 0
    );
    if !bare {
        return;
    }
    let at = graph.node(id).map(|n| n.pos).unwrap_or(Pos2::ZERO);
    let r = graph.add(
        NodeKind::Recombine(RecombineParams { n: 1, pair: id, ..Default::default() }),
        at + egui::Vec2::new(0.0, DROP),
    );
    if let Some(NodeKind::Breakout(p)) = graph.node_mut(id).map(|k| &mut k.kind) {
        p.pair = r;
    }
    graph.connect_out(r, 0, id, 0);
    sync(graph);
}

/// Re-point every consumer of a Breakout output at that output's hidden
/// kid, move any viewer slot parked on one, and drop the Breakouts — the
/// pre-pass `to_engine_graph_at_f` runs so the engine only ever sees
/// ordinary ChannelSelects. `None` when there is nothing to do (the common
/// case, and what makes the recursion one level deep).
pub fn expand(g: &EditorGraph) -> Option<EditorGraph> {
    if !g.nodes.values().any(|n| matches!(n.kind, NodeKind::Breakout(_))) {
        return None;
    }
    let mut out = g.clone();
    let breakouts: Vec<u64> = out
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n.kind, NodeKind::Breakout(_)))
        .map(|(id, _)| *id)
        .collect();
    for b in breakouts {
        let (kids, remainder, rkids, pair) = match out.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => {
                (p.kids.clone(), p.remainder, p.remainder_kids.clone(), p.pair)
            }
            _ => continue,
        };
        // The receiver's alpha: its porch input takes the BASE RGBA kid,
        // wired here rather than in the editor graph — the kid is hidden,
        // and a wire out of a node nobody can see would draw from nowhere.
        // A hand-wired porch wins: this only fills an empty one.
        if pair != 0 {
            let want = match out.node(pair).map(|n| &n.kind) {
                Some(NodeKind::Recombine(rp))
                    if rp.alpha == engine::graph::AlphaFrom::Base =>
                {
                    Some(rp.base_input())
                }
                _ => None,
            };
            if let (Some(port), Some(base)) = (want, rkids.first().copied()) {
                if out.input_of(pair, port).is_none() {
                    out.connect(pair, port, base);
                }
            }
        }
        let kid_for = |k: u32| -> Option<u64> {
            // The string past the layers is the remainder (its signed
            // sum); past the end otherwise reads as the last string, the
            // lumabands rule — an edge left by a layer edit gives a
            // picture, not a black frame.
            if remainder && k as usize >= kids.len() {
                return rkids.get(1).copied();
            }
            kids.get((k as usize).min(kids.len().saturating_sub(1))).copied()
        };
        let consumers: Vec<(u64, usize, u32)> = out
            .edges
            .iter()
            .filter(|(_, _, s)| *s == b)
            .map(|(d, i, _)| (*d, *i, out.out_of(*d, *i)))
            .collect();
        for (d, i, k) in consumers {
            match kid_for(k) {
                Some(kid) => out.connect(d, i, kid),
                None => out.disconnect_input(d, i),
            }
        }
        // A viewer slot parked on the node shows the string it named.
        let park = |v: &mut Option<u64>, vo: &mut u32| {
            if *v == Some(b) {
                *v = kid_for(*vo);
                *vo = 0;
            }
        };
        let (mut v, mut vo) = (out.view, out.view_out);
        park(&mut v, &mut vo);
        (out.view, out.view_out) = (v, vo);
        let (mut v, mut vo) = (out.view_b, out.view_b_out);
        park(&mut v, &mut vo);
        (out.view_b, out.view_b_out) = (v, vo);
        let (mut v, mut vo) = (out.view_c, out.view_c_out);
        park(&mut v, &mut vo);
        (out.view_c, out.view_c_out) = (v, vo);
        let (mut v, mut vo) = (out.view_d, out.view_d_out);
        park(&mut v, &mut vo);
        (out.view_d, out.view_d_out) = (v, vo);
        out.nodes.remove(&b);
        out.edges.retain(|(d, _, s)| *d != b && *s != b);
        out.edge_out.retain(|(d, _), _| *d != b);
        out.hidden_wires.retain(|(d, _)| *d != b);
    }
    Some(out)
}

/// The prefix a layer groups under: everything before the first
/// separator. `light1.R`-style channel suffixes are gone by the time a
/// LAYER list exists, so this is about pass families: `light_key`,
/// `light_fill` share `light`; `diffuse` stands alone.
fn prefix(name: &str) -> &str {
    name.split(['.', '_', '-']).next().unwrap_or(name)
}

fn is_base(name: &str) -> bool {
    name.is_empty() || name.eq_ignore_ascii_case("rgba")
}

/// Guess the LIGHT-GROUP layers: names sharing a prefix with at least one
/// other layer (the base rgba never counts). `lights[i]` answers for
/// `names[i]`.
pub fn guess_lights(names: &[String]) -> Vec<bool> {
    names
        .iter()
        .map(|n| {
            !is_base(n)
                && names.iter().any(|m| {
                    !is_base(m) && !std::ptr::eq(n, m) && n != m && prefix(n) == prefix(m)
                })
        })
        .collect()
}

/// Guess the plain AOVs: everything that is NOT a light group and not the
/// base rgba — the loners (diffuse, spec, gi, depth).
pub fn guess_aovs(names: &[String]) -> Vec<bool> {
    let lights = guess_lights(names);
    names
        .iter()
        .zip(lights)
        .map(|(n, lit)| !is_base(n) && !lit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::MultParams;

    fn multi_read(g: &mut EditorGraph) -> u64 {
        let id = g.add(NodeKind::read(0, "plate.exr"), egui::pos2(0.0, -200.0));
        if let Some(NodeKind::Read { multi, .. }) = g.node_mut(id).map(|n| &mut n.kind) {
            *multi = true;
        }
        id
    }

    #[test]
    fn the_harp_lowers_to_selects_and_a_fused_sum() {
        let mut g = EditorGraph::default();
        let rd = multi_read(&mut g);
        let (b, r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into(), "gi".into()],
        );
        g.connect(b, 0, rd);
        sync(&mut g);
        // A grade spliced into string 1, the harp's whole point.
        let grade = g.add(NodeKind::Mult(MultParams::default()), egui::pos2(0.0, 150.0));
        g.connect_out(grade, 0, b, 1);
        g.connect(r, 1, grade);
        let kids = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => p.kids.clone(),
            _ => panic!("breakout lost"),
        };
        assert_eq!(kids.len(), 3);
        // Kids ride the node's own cable.
        for k in &kids {
            assert_eq!(g.input_of(*k, 0), Some(rd));
        }
        // The app's rebuild pass mints kid slots; stand in for it here.
        for (i, k) in kids.iter().enumerate() {
            if let Some(NodeKind::ChannelSelect(p)) =
                g.node_mut(*k).map(|n| &mut n.kind)
            {
                p.source = 10 + i;
            }
        }
        let eg = g.to_engine_graph();
        // The Breakout is gone; the Recombine's strings come from the kids
        // (string 1 through the spliced grade), and each kid lowered to a
        // Read of its layer.
        assert!(!eg.nodes.values().any(|k| matches!(k, NodeKind::Breakout(_))));
        assert_eq!(eg.edges.get(&(r, 0)).copied(), Some(kids[0]));
        assert_eq!(eg.edges.get(&(grade, 0)).copied(), Some(kids[1]));
        assert_eq!(eg.edges.get(&(r, 1)).copied(), Some(grade));
        assert_eq!(eg.edges.get(&(r, 2)).copied(), Some(kids[2]));
        for (i, k) in kids.iter().enumerate() {
            match eg.nodes.get(k) {
                Some(NodeKind::Read { layer: Some(l), .. }) => {
                    assert_eq!(l, [&"diff", &"spec", &"gi"][i]);
                }
                other => panic!("kid {i} lowered to {other:?}"),
            }
        }
    }

    #[test]
    fn layer_edits_grow_and_shrink_the_kids_and_the_receiver() {
        let mut g = EditorGraph::default();
        let rd = multi_read(&mut g);
        let (b, r) = spawn_pair(&mut g, egui::pos2(0.0, 0.0), vec!["diff".into()]);
        g.connect(b, 0, rd);
        sync(&mut g);
        if let Some(NodeKind::Breakout(p)) = g.node_mut(b).map(|n| &mut n.kind) {
            p.layers.push("spec".into());
        }
        sync(&mut g);
        let (kids, n) = match (g.node(b).map(|n| &n.kind), g.node(r).map(|n| &n.kind)) {
            (Some(NodeKind::Breakout(p)), Some(NodeKind::Recombine(q))) => {
                (p.kids.clone(), q.n)
            }
            _ => panic!("pair lost"),
        };
        assert_eq!((kids.len(), n), (2, 2));
        // The grown string wired itself down; a cut string stays cut.
        assert_eq!(g.input_of(r, 1), Some(b));
        g.disconnect_input(r, 0);
        if let Some(NodeKind::Breakout(p)) = g.node_mut(b).map(|n| &mut n.kind) {
            p.layers.push("gi".into());
        }
        sync(&mut g);
        assert_eq!(g.input_of(r, 0), None, "a hand-cut string must stay cut");
        assert_eq!(g.input_of(r, 2), Some(b));
        if let Some(NodeKind::Breakout(p)) = g.node_mut(b).map(|n| &mut n.kind) {
            p.layers.truncate(1);
        }
        sync(&mut g);
        let strings = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => p.kids.len(),
            _ => panic!("breakout lost"),
        };
        assert_eq!(strings, 1, "one string, one kid");
        // The base-rgba kid is hidden machinery too, but it belongs to the
        // receiver's ALPHA, not to a layer.
        assert_eq!(kid_set(&g).len(), 2);
    }

    #[test]
    fn a_string_re_points_in_place_keeping_its_kid_and_slot() {
        let mut g = EditorGraph::default();
        let rd = multi_read(&mut g);
        let (b, r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into()],
        );
        g.connect(b, 0, rd);
        sync(&mut g);
        let kids = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => p.kids.clone(),
            _ => panic!("breakout lost"),
        };
        // Mint, then splice a grade into string 1 — the chain that must
        // survive the re-point.
        for (i, k) in kids.iter().enumerate() {
            if let Some(NodeKind::ChannelSelect(p)) =
                g.node_mut(*k).map(|n| &mut n.kind)
            {
                p.source = 20 + i;
            }
        }
        let grade = g.add(NodeKind::Mult(MultParams::default()), egui::pos2(0.0, 150.0));
        g.connect_out(grade, 0, b, 1);
        g.connect(r, 1, grade);
        // The after-the-fact edit: string 1 re-aims spec -> gi.
        if let Some(NodeKind::Breakout(p)) = g.node_mut(b).map(|n| &mut n.kind) {
            p.layers[1] = "gi".into();
        }
        sync(&mut g);
        let (kids2, layer, slot) = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => {
                match g.node(p.kids[1]).map(|n| &n.kind) {
                    Some(NodeKind::ChannelSelect(cs)) => {
                        (p.kids.clone(), cs.layer.clone(), cs.source)
                    }
                    _ => panic!("kid lost"),
                }
            }
            _ => panic!("breakout lost"),
        };
        assert_eq!(kids2, kids, "re-pointing must keep the kids");
        assert_eq!(layer, "gi");
        assert_eq!(slot, 21, "the kid keeps its minted slot");
        // The spliced grade still sits in the string.
        assert_eq!(g.input_of(grade, 0), Some(b));
        assert_eq!(g.input_of(r, 1), Some(grade));
        let eg = g.to_engine_graph();
        match eg.nodes.get(&kids[1]) {
            Some(NodeKind::Read { layer: Some(l), .. }) => assert_eq!(l, "gi"),
            other => panic!("kid lowered to {other:?}"),
        }
    }

    /// Both bookends live UNDER the slab: the base rgba goes in on the
    /// left, the rebuilt beauty leaves on the right, and neither socket
    /// may hang off the end of the bar you can grab.
    /// A string moved about INSIDE the comb leaves the bar alone; only one
    /// that reaches past an end takes the bar with it. The old sizing grew
    /// the slab with every nudge and the reorder snapped it back — a drag
    /// that ratcheted.
    #[test]
    fn the_bar_holds_still_until_a_string_passes_its_end() {
        let mut g = EditorGraph::default();
        let (b, _r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into(), "gi".into()],
        );
        let spacing = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => p.spacing,
            _ => panic!("breakout lost"),
        };
        let rect = |gr: &EditorGraph| wide_rect(&gr.node(b).unwrap().kind).unwrap();
        let (half0, dx0) = rect(&g);
        // The MIDDLE string, anywhere short of its neighbours: nothing.
        for d in [0.2, 0.5, 0.9, -0.9] {
            nudge(&mut g, b, 1, d * spacing);
            let (half, dx) = rect(&g);
            assert!(
                (half.x - half0.x).abs() < 1e-3 && (dx - dx0).abs() < 1e-3,
                "the bar moved for a middle string at {d} pitch: {half:?} {dx}"
            );
            nudge(&mut g, b, 1, -d * spacing);
        }
        // The LAST string, reaching out past the end: the bar follows it.
        nudge(&mut g, b, 2, 0.5 * spacing);
        let (half, dx) = rect(&g);
        assert!((half.x - (half0.x + 0.25 * spacing)).abs() < 1e-3, "half {half:?}");
        assert!((dx - (dx0 + 0.25 * spacing)).abs() < 1e-3, "centre {dx}");
        nudge(&mut g, b, 2, -0.5 * spacing);
        // And inward, it shrinks to hug them.
        nudge(&mut g, b, 0, 0.5 * spacing);
        assert!(rect(&g).0.x < half0.x);
    }

    /// A chip dragged past its neighbour CHANGES PLACES with it, and the
    /// string takes everything of its own along: its layer, its kid, the
    /// chain spliced into it, its gain and its keyed gain. Nothing moves
    /// on screen, and the nudge that carried it there is given back — so
    /// the slab stays the width of its comb.
    #[test]
    fn a_string_dragged_past_its_neighbour_changes_places() {
        let mut g = EditorGraph::default();
        let rd = multi_read(&mut g);
        let (b, r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into(), "gi".into()],
        );
        g.connect(b, 0, rd);
        sync(&mut g);
        let (kids, spacing) = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => (p.kids.clone(), p.spacing),
            _ => panic!("breakout lost"),
        };
        // A grade spliced into string 0, and gains that name their string.
        let grade = g.add(NodeKind::Mult(MultParams::default()), egui::pos2(0.0, 150.0));
        g.connect_out(grade, 0, b, 0);
        g.connect(r, 0, grade);
        let mut curve = crate::anim::AnimCurve::default();
        curve.set_key(7, 0.25);
        if let Some(rn) = g.node_mut(r) {
            if let NodeKind::Recombine(p) = &mut rn.kind {
                p.gain = vec![0.25, 0.5, 0.75];
            }
            rn.anim.insert(crate::anim::Channel::RecombineGain(0), curve);
        }
        // Drag string 0 a pitch and a bit to the right, the way the app
        // does, then let the bar settle.
        let before = |gr: &EditorGraph, i: usize| -> f32 {
            match gr.node(b).map(|n| &n.kind) {
                Some(NodeKind::Breakout(p)) => {
                    i as f32 * p.spacing + p.offsets.get(i).copied().unwrap_or(0.0)
                }
                _ => f32::NAN,
            }
        };
        let (u0, u1) = (before(&g, 0), before(&g, 1));
        nudge(&mut g, b, 0, spacing * 1.1);
        let landed = pass_reorder(&mut g, b, 0);
        assert_eq!(landed, 1, "it passed exactly one neighbour");
        let (layers, kids2, offs) = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => {
                (p.layers.clone(), p.kids.clone(), p.offsets.clone())
            }
            _ => panic!("breakout lost"),
        };
        assert_eq!(layers, vec!["spec", "diff", "gi"], "the order follows the drag");
        assert_eq!(kids2, vec![kids[1], kids[0], kids[2]], "each kid went with its layer");
        // The dragged string stayed under the cursor; the neighbour
        // shuffled aside into the slot it vacated.
        assert!((before(&g, 1) - (u0 + spacing * 1.1)).abs() < 1e-3);
        assert!((before(&g, 0) - u0).abs() < 1e-3, "displaced into the free slot");
        let _ = u1;
        assert!(
            offs.iter().all(|o| o.abs() < spacing),
            "a swapped string keeps its nudge under one pitch, got {offs:?}"
        );
        // The spliced chain came with it: the grade still reads its own
        // layer's nodule and still lands in its own string.
        assert_eq!(g.out_of(grade, 0), 1);
        assert_eq!(g.input_of(r, 1), Some(grade));
        assert_eq!(g.input_of(r, 0), Some(b), "the displaced string is plain");
        assert_eq!(g.out_of(r, 0), 0);
        // Gains and keys are the string's, not the slot's.
        match g.node(r).map(|n| &n.kind) {
            Some(NodeKind::Recombine(p)) => assert_eq!(p.gain, vec![0.5, 0.25, 0.75]),
            _ => panic!("receiver lost"),
        }
        let anim = &g.node(r).unwrap().anim;
        assert!(anim.contains_key(&crate::anim::Channel::RecombineGain(1)));
        assert!(!anim.contains_key(&crate::anim::Channel::RecombineGain(0)));
        // And the engine still sees the right layer per string.
        for (i, k) in kids2.iter().enumerate() {
            if let Some(NodeKind::ChannelSelect(p)) = g.node_mut(*k).map(|n| &mut n.kind)
            {
                p.source = 40 + i;
            }
        }
        let eg = g.to_engine_graph();
        assert_eq!(eg.edges.get(&(grade, 0)).copied(), Some(kids2[1]));
        match eg.nodes.get(&kids2[0]) {
            Some(NodeKind::Read { layer: Some(l), .. }) => assert_eq!(l, "spec"),
            other => panic!("string 0 lowered to {other:?}"),
        }
    }

    #[test]
    fn the_receivers_bookends_sit_inside_its_slab() {
        let mut g = EditorGraph::default();
        let (_b, r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into(), "gi".into()],
        );
        let n = g.node(r).unwrap();
        let (half, dx) = wide_rect(&n.kind).expect("the receiver is a wide bar");
        let (lo, hi) = (n.pos.x + dx - half.x, n.pos.x + dx + half.x);
        let base = base_x(&n.kind, n.pos, 1.0).expect("a base porch");
        let out = end_x(&n.kind, n.pos, 1.0).expect("an output porch");
        let s0 = string_x(&n.kind, n.pos, 0, 1.0).unwrap();
        assert!(base < s0, "the base porch is left of the first string");
        assert!(out > string_x(&n.kind, n.pos, 2, 1.0).unwrap(), "output on the right");
        assert!(lo <= base && base <= hi, "base {base} outside the slab {lo}..{hi}");
        assert!(lo <= out && out <= hi, "output {out} outside the slab {lo}..{hi}");
        // The strings stay centred on the node between the two bookends.
        assert!((string_x(&n.kind, n.pos, 1, 1.0).unwrap() - n.pos.x).abs() < 1e-3);
    }

    #[test]
    fn the_receiver_rides_x_locked_under_the_breakout() {
        let mut g = EditorGraph::default();
        let (b, r) = spawn_pair(&mut g, egui::pos2(10.0, 0.0), vec!["diff".into()]);
        if let Some(n) = g.node_mut(r) {
            n.pos = egui::pos2(400.0, 500.0); // a sideways drag
        }
        if let Some(n) = g.node_mut(b) {
            n.pos.x = -60.0;
        }
        sync(&mut g);
        let (bx, rp) = (g.node(b).unwrap().pos.x, g.node(r).unwrap().pos);
        assert_eq!(bx, -60.0);
        assert_eq!(rp.x, -60.0, "x snaps to the breakout");
        assert_eq!(rp.y, 500.0, "y stays where the hand put it");
    }

    #[test]
    fn the_top_bar_carries_its_spliced_strings_but_not_the_receiver() {
        let mut g = EditorGraph::default();
        let rd = multi_read(&mut g);
        let (b, r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into(), "gi".into()],
        );
        g.connect(b, 0, rd);
        sync(&mut g);
        // Two nodes in series on string 1, one on string 2.
        let a = g.add(NodeKind::Mult(MultParams::default()), egui::pos2(0.0, 100.0));
        let c = g.add(NodeKind::Mult(MultParams::default()), egui::pos2(0.0, 200.0));
        let d = g.add(NodeKind::Mult(MultParams::default()), egui::pos2(40.0, 150.0));
        g.connect_out(a, 0, b, 1);
        g.connect(c, 0, a);
        g.connect(r, 1, c);
        g.connect_out(d, 0, b, 2);
        g.connect(r, 2, d);
        let mut got = riders(&g, &[b]);
        got.sort();
        let mut want = vec![a, c, d];
        want.sort();
        assert_eq!(got, want);
        // Riders never include what is already in hand, the read above or
        // the receiver below; and the receiver alone carries nothing.
        assert_eq!(riders(&g, &[b, a]).len(), 2);
        assert!(!riders(&g, &[b]).contains(&r) && !riders(&g, &[b]).contains(&rd));
        assert!(riders(&g, &[r]).is_empty());
    }

    #[test]
    fn light_and_aov_guesses_split_on_shared_prefixes() {
        let names: Vec<String> = [
            "", "diffuse", "spec", "light_key", "light_fill", "lgt.rim", "depth",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let lights = guess_lights(&names);
        let aovs = guess_aovs(&names);
        assert_eq!(lights, [false, false, false, true, true, false, false]);
        assert_eq!(aovs, [false, true, true, false, false, true, true]);
        // rgba (the empty base) lands in neither.
        assert!(!lights[0] && !aovs[0]);
    }

    /// The rebuild's alpha: the receiver's porch takes a hidden base-rgba
    /// kid off the same cable the strings come from, wired in the LOWERING
    /// (the kid is invisible, so the editor graph must stay clean), and
    /// only while the mode asks for it.
    #[test]
    fn the_receiver_takes_its_alpha_off_the_base_rgba() {
        use engine::graph::AlphaFrom;
        let mut g = EditorGraph::default();
        let rd = multi_read(&mut g);
        let (b, r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into()],
        );
        g.connect(b, 0, rd);
        sync(&mut g);
        // Base is the default, so a base kid exists and rides the cable.
        let base = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => {
                assert_eq!(p.remainder_kids.len(), 1, "just the base kid");
                p.remainder_kids[0]
            }
            _ => panic!("breakout lost"),
        };
        assert!(matches!(
            g.node(base).map(|n| &n.kind),
            Some(NodeKind::ChannelSelect(p)) if p.layer.is_empty()
        ));
        assert_eq!(g.input_of(base, 0), Some(rd));
        // The editor graph leaves the porch EMPTY; the lowering fills it.
        let port = match g.node(r).map(|n| &n.kind) {
            Some(NodeKind::Recombine(p)) => p.base_input(),
            _ => panic!("receiver lost"),
        };
        assert_eq!(g.input_of(r, port), None, "no wire out of a hidden node");
        if let Some(NodeKind::ChannelSelect(p)) = g.node_mut(base).map(|n| &mut n.kind) {
            p.source = 30;
        }
        let eg = g.to_engine_graph();
        assert_eq!(eg.edges.get(&(r, port)).copied(), Some(base));
        // Summed: no kid, no porch wire.
        if let Some(NodeKind::Recombine(p)) = g.node_mut(r).map(|n| &mut n.kind) {
            p.alpha = AlphaFrom::Sum;
        }
        sync(&mut g);
        assert!(g.node(base).is_none(), "the base kid goes with the mode");
        let eg = g.to_engine_graph();
        assert_eq!(eg.edges.get(&(r, port)).copied(), None);
        // A HAND-WIRED porch always wins.
        if let Some(NodeKind::Recombine(p)) = g.node_mut(r).map(|n| &mut n.kind) {
            p.alpha = AlphaFrom::Base;
        }
        sync(&mut g);
        let mine = g.add(NodeKind::Mult(MultParams::default()), egui::pos2(-400.0, 0.0));
        g.connect(r, port, mine);
        let eg = g.to_engine_graph();
        assert_eq!(eg.edges.get(&(r, port)).copied(), Some(mine));
    }

    #[test]
    fn the_remainder_is_the_beauty_minus_every_string() {
        let mut g = EditorGraph::default();
        let rd = multi_read(&mut g);
        let (b, r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into()],
        );
        g.connect(b, 0, rd);
        if let Some(NodeKind::Breakout(p)) = g.node_mut(b).map(|n| &mut n.kind) {
            p.remainder = true;
        }
        sync(&mut g);
        let (kids, rkids) = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => (p.kids.clone(), p.remainder_kids.clone()),
            _ => panic!("breakout lost"),
        };
        assert_eq!(rkids.len(), 2);
        let (base, rsum) = (rkids[0], rkids[1]);
        // The base select reads the plain rgba off the node's own cable.
        match g.node(base).map(|n| &n.kind) {
            Some(NodeKind::ChannelSelect(p)) => assert!(p.layer.is_empty()),
            other => panic!("base is {other:?}"),
        }
        assert_eq!(g.input_of(base, 0), Some(rd));
        // The signed sum: base plus, every layer kid minus.
        match g.node(rsum).map(|n| &n.kind) {
            Some(NodeKind::Recombine(p)) => {
                assert_eq!(p.n, 3);
                assert_eq!(p.sub, vec![false, true, true]);
            }
            other => panic!("rsum is {other:?}"),
        }
        assert_eq!(g.input_of(rsum, 0), Some(base));
        assert_eq!(g.input_of(rsum, 1), Some(kids[0]));
        assert_eq!(g.input_of(rsum, 2), Some(kids[1]));
        // The receiver grew a third string, wired to the remainder output,
        // and the lowering hands that string the signed sum.
        match g.node(r).map(|n| &n.kind) {
            Some(NodeKind::Recombine(p)) => assert_eq!(p.n, 3),
            _ => panic!("receiver lost"),
        }
        assert_eq!(g.input_of(r, 2), Some(b));
        let eg = g.to_engine_graph();
        assert_eq!(eg.edges.get(&(r, 2)).copied(), Some(rsum));
        // Turning it off takes the remainder's own machinery with it. The
        // base-rgba kid STAYS: the receiver's alpha is still coming off it.
        if let Some(NodeKind::Breakout(p)) = g.node_mut(b).map(|n| &mut n.kind) {
            p.remainder = false;
        }
        sync(&mut g);
        assert!(g.node(rsum).is_none(), "the signed sum goes");
        assert!(g.node(base).is_some(), "the alpha still needs the beauty");
        // With nothing left wanting it, it goes too.
        if let Some(NodeKind::Recombine(p)) = g.node_mut(r).map(|n| &mut n.kind) {
            p.alpha = engine::graph::AlphaFrom::Sum;
        }
        sync(&mut g);
        assert!(g.node(base).is_none());
    }

    #[test]
    fn a_viewed_string_survives_the_lowering() {
        let mut g = EditorGraph::default();
        let rd = multi_read(&mut g);
        let (b, _r) = spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into()],
        );
        g.connect(b, 0, rd);
        sync(&mut g);
        g.view = Some(b);
        g.view_out = 1;
        let eg = g.to_engine_graph();
        let kids = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => p.kids.clone(),
            _ => unreachable!(),
        };
        assert_eq!(eg.view, Some(kids[1]));
    }
}


/// The spliced chain on string `k`, top to bottom — every node between
/// the bar's output and the receiver's input. Empty when the string is
/// bare; None when it does not reach the bar (broken elsewhere).
pub fn string_chain(
    graph: &EditorGraph,
    bar: u64,
    rec: u64,
    k: usize,
) -> Option<Vec<u64>> {
    let mut chain: Vec<u64> = Vec::new();
    let mut at = graph.edges.iter().find(|(d, i, _)| *d == rec && *i == k).map(|e| e.2);
    for _ in 0..32 {
        let id = at?;
        if id == bar {
            chain.reverse();
            return Some(chain);
        }
        chain.push(id);
        at = graph.edges.iter().find(|(d, i, _)| *d == id && *i == 0).map(|e| e.2);
    }
    None
}

/// The nodes spliced into the strings of every harp bar in `ids` — what
/// rides along when the top bar is dragged, so a bar moved up or down
/// carries its graded strings with it instead of leaving them behind.
/// The receiver is NOT a rider: it holds its height (and is x-locked
/// anyway), so moving the top bar lengthens or shortens the strings.
pub fn riders(graph: &EditorGraph, ids: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    for id in ids {
        let Some(NodeKind::Breakout(p)) = graph.node(*id).map(|n| &n.kind) else { continue };
        if p.pair == 0 {
            continue;
        }
        for k in 0..p.strings() {
            for n in string_chain(graph, *id, p.pair, k).unwrap_or_default() {
                if !ids.contains(&n) && !out.contains(&n) {
                    out.push(n);
                }
            }
        }
    }
    out
}

/// Put every harp back on its strings: bars x-locked and in order, each
/// string's spliced chain AT its string x, spaced evenly between the bars.
/// The generic auto-layout has no idea a harp is an instrument — it scores
/// wire length and happily weaves the strings into a cat's cradle — so
/// `tidy` calls this afterwards, and running tidy on a woven harp UNWEAVES
/// it rather than making it worse.
pub fn repair(graph: &mut EditorGraph) {
    let bars: Vec<(u64, u64)> = graph
        .nodes
        .iter()
        .filter_map(|(id, n)| match &n.kind {
            NodeKind::Breakout(p) if p.pair != 0 => Some((*id, p.pair)),
            _ => None,
        })
        .collect();
    for (b, r) in bars {
        let (Some(bn), Some(rn)) = (graph.node(b), graph.node(r)) else { continue };
        let (bpos, bsize, bkind) = (bn.pos, bn.size, bn.kind.clone());
        let strings = match &bkind {
            NodeKind::Breakout(p) => p.strings(),
            _ => continue,
        };
        // Bars: same column, receiver below.
        let mut rpos = rn.pos;
        rpos.x = bpos.x;
        if rpos.y < bpos.y + 80.0 {
            rpos.y = bpos.y + 240.0;
        }
        if let Some(n) = graph.node_mut(r) {
            n.pos = rpos;
        }
        for k in 0..strings {
            let Some(x) = string_x(&bkind, bpos, k, bsize) else { continue };
            let Some(chain) = string_chain(graph, b, r, k) else {
                continue; // string broken elsewhere; not ours to move
            };
            let n = chain.len() as f32;
            for (i, id) in chain.iter().enumerate() {
                if let Some(node) = graph.node_mut(*id) {
                    node.pos.x = x;
                    node.pos.y =
                        bpos.y + (rpos.y - bpos.y) * ((i as f32 + 1.0) / (n + 1.0));
                }
            }
        }
    }
}
