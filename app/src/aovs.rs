//! AOV rebuild: Wobble's answer to the python snippet every compositor keeps in
//! a menu — point it at a multi-layer EXR Read, tick the layers you want,
//! and it builds the rig: the Read flips to its all-channels (fat cable)
//! output, ONE ChannelSelect per ticked layer hangs off it, and the selects
//! sum through a Blend(Add) ladder that reconstructs the beauty. Each
//! layer's branch is its own hot wire — exactly where a Grade goes — and
//! only ticked layers ever decode or cache (a select lowers to its own
//! reader). Consumers of the original Read move onto the rebuilt sum.
//!
//! The graph surgery is a pure function here (unit-testable); the dialog
//! lives in lib.rs.

use crate::editor::EditorGraph;
use engine::graph::{BlendOp, ChannelSelectParams, NodeKind};

/// Build the rig for `picked` layer names (empty string = base rgba).
/// `selects[k]` picks `picked[k]`; the last id in `adds` (or the lone
/// select) is the rebuilt beauty. `next_slot` is the app's reader-slot
/// counter — each select is a reader.
pub struct Rig {
    pub selects: Vec<u64>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub adds: Vec<u64>,
}

pub fn build_rig(
    graph: &mut EditorGraph,
    src: u64,
    next_slot: &mut usize,
    picked: &[String],
) -> Option<Rig> {
    if picked.is_empty() {
        return None;
    }
    let pos = graph.node(src)?.pos;
    {
        let n = graph.node_mut(src)?;
        let NodeKind::Read { multi, .. } = &mut n.kind else { return None };
        // The source becomes the one fat cable everything hangs off.
        *multi = true;
    }
    // One column per layer, ladder of Adds descending to the right — the
    // classic comb, so every branch has room for its grade.
    let (x0, y0) = (pos.x - (picked.len() as f32 - 1.0) * 95.0, pos.y + 120.0);
    let mut selects = Vec::new();
    let mut adds = Vec::new();
    let mut sum: Option<u64> = None;
    for (i, name) in picked.iter().enumerate() {
        let slot = *next_slot;
        *next_slot += 1;
        let cs = graph.add(
            NodeKind::ChannelSelect(ChannelSelectParams {
                source: slot,
                layer: name.clone(),
            }),
            egui::pos2(x0 + i as f32 * 190.0, y0),
        );
        graph.connect(cs, 0, src);
        selects.push(cs);
        sum = Some(match sum {
            None => cs,
            Some(prev) => {
                let add = graph.add(
                    NodeKind::Blend { op: BlendOp::Add, mix: 1.0, gain: [1.0, 1.0] },
                    egui::pos2(x0 + i as f32 * 190.0, y0 + 90.0 + i as f32 * 70.0),
                );
                // Input 0 (foreground) = this layer's branch — the wire a
                // Grade drops onto; input 1 = the running sum.
                graph.connect(add, 0, cs);
                graph.connect(add, 1, prev);
                adds.push(add);
                add
            }
        });
    }
    let beauty = sum.expect("picked is non-empty");
    // The rebuilt beauty takes over the original Read's consumers.
    let consumers: Vec<(u64, usize)> = graph
        .edges
        .iter()
        .filter(|(d, _, s)| *s == src && !selects.contains(d) && !adds.contains(d))
        .map(|(d, i, _)| (*d, *i))
        .collect();
    for (d, i) in consumers {
        graph.connect(d, i, beauty);
    }
    Some(Rig { selects, adds })
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::Pos2;

    fn read_graph() -> (EditorGraph, u64, u64) {
        let mut g = EditorGraph::default();
        let read = g.add(NodeKind::read(0, "/x/multi.exr"), Pos2::new(0.0, 0.0));
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 140.0));
        g.connect(grade, 0, read);
        (g, read, grade)
    }

    #[test]
    fn a_rig_is_one_read_fanning_into_selects_and_adds() {
        let (mut g, read, grade) = read_graph();
        let mut slot = 1;
        let picked =
            ["diffuse".to_string(), "specular".to_string(), "sss".to_string()];
        let rig = build_rig(&mut g, read, &mut slot, &picked).expect("built");
        assert_eq!(rig.selects.len(), 3);
        assert_eq!(rig.adds.len(), 2, "n layers sum through n-1 adds");
        assert_eq!(slot, 4, "each select is its own reader");
        // The source flipped to the fat cable; ONE read total.
        assert!(matches!(
            g.node(read).unwrap().kind,
            NodeKind::Read { multi: true, .. }
        ));
        // Every select hangs off the read, carries its layer, and LOWERS to
        // a Read of that layer through its own slot.
        for (cs, want) in rig.selects.iter().zip(&picked) {
            assert_eq!(g.channel_source(*cs), Some(read));
            let eng = g.to_engine_graph();
            let NodeKind::Read { path, layer, multi, .. } = &eng.nodes[cs] else {
                panic!("select did not lower to a read");
            };
            assert_eq!(path, "/x/multi.exr");
            assert_eq!(layer.as_deref(), Some(want.as_str()));
            assert!(!multi);
        }
        // The adds ladder: each add takes a branch (0) and the sum (1).
        let beauty = *rig.adds.last().unwrap();
        assert!(matches!(
            g.node(rig.adds[0]).unwrap().kind,
            NodeKind::Blend { op: BlendOp::Add, .. }
        ));
        // The old consumer now reads the rebuilt beauty.
        assert_eq!(
            g.edges.iter().find(|(d, i, _)| (*d, *i) == (grade, 0)).unwrap().2,
            beauty
        );
    }

    #[test]
    fn the_base_layer_lowers_as_no_layer() {
        let (mut g, read, _) = read_graph();
        let mut slot = 1;
        let rig = build_rig(&mut g, read, &mut slot, &["".to_string()]).unwrap();
        let eng = g.to_engine_graph();
        let NodeKind::Read { layer, .. } = &eng.nodes[&rig.selects[0]] else {
            panic!();
        };
        assert!(layer.is_none());
        assert!(rig.adds.is_empty(), "one layer needs no add");
    }

    #[test]
    fn a_non_read_refuses() {
        let (mut g, _, grade) = read_graph();
        let mut slot = 1;
        assert!(build_rig(&mut g, grade, &mut slot, &["a".into()]).is_none());
    }
}
