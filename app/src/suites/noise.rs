//! `--test-noise`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-noise driver: the whole authoring loop for a noise field —
    /// dive into a Noise node, build a graph inside it, step back out, and
    /// check that what was authored is what renders. The conversions have
    /// their own unit tests; this is the machinery around them.
    pub(crate) fn drive_test_noise(&mut self) {
        let Some((stage, mut frames, nid)) = self.test_noise else { return };
        frames += 1;
        if frames > 900 {
            println!("NOISE TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let id = self
                    .egraph
                    .add(NodeKind::Noise(Default::default()), egui::Pos2::new(320.0, 40.0));
                self.selected = vec![id];
                if let Err(e) = self.dive_into(id) {
                    println!("NOISE TEST FAIL: could not open the node: {e}");
                    std::process::exit(1);
                }
                self.test_noise = Some((1, 0, id));
                return;
            }
            // Inside: build a field whose fBm scale is driven by a Coord,
            // which is the paradigm in its smallest honest form.
            1 if frames > 2 => {
                if self.dive.last().map(|d| d.kind) != Some(DiveKind::Noise) {
                    println!("NOISE TEST FAIL: the dive did not open a field graph");
                    std::process::exit(1);
                }
                use engine::field::{CoordKind, FieldNode};
                let c = self.egraph.add(
                    NodeKind::Field(FieldNode::Coord { kind: CoordKind::X, scale: 1.0 }),
                    egui::Pos2::new(0.0, 0.0),
                );
                let r = self.egraph.add(
                    NodeKind::Field(FieldNode::Remap {
                        in_lo: 0.0,
                        in_hi: 1.0,
                        out_lo: 0.4,
                        out_hi: 0.03,
                        clamp: true,
                    }),
                    egui::Pos2::new(0.0, 120.0),
                );
                let f = self.egraph.add(
                    NodeKind::Field(FieldNode::default()),
                    egui::Pos2::new(0.0, 240.0),
                );
                self.egraph.edges.push((r, 0, c));
                self.egraph.edges.push((f, 0, r));
                self.egraph.edges.push((f, 1, r));
                // Opening an empty Noise seeds an Output; wire the chain
                // into it, which is the authoring gesture this is here to
                // test.
                let out = self.egraph.nodes.iter().find_map(|(id, n)| {
                    matches!(n.kind, NodeKind::Field(engine::field::FieldNode::Output))
                        .then_some(*id)
                });
                match out {
                    Some(o) => self.egraph.edges.push((o, 0, f)),
                    None => {
                        println!("NOISE TEST FAIL: an empty dive did not seed an Output");
                        std::process::exit(1);
                    }
                }
                next = 2;
            }
            2 if frames > 2 => {
                if let Err(e) = self.dive_out() {
                    println!("NOISE TEST FAIL: could not close the node: {e}");
                    std::process::exit(1);
                }
                next = 3;
            }
            3 if frames > 2 => {
                // What came back out has to be what was built inside.
                let n = match self.egraph.node(nid).map(|e| &e.kind) {
                    Some(NodeKind::Noise(n)) => n.clone(),
                    _ => {
                        println!("NOISE TEST FAIL: the Noise node is gone");
                        std::process::exit(1);
                    }
                };
                if n.graph.nodes.len() != 4 || n.graph.edges.len() != 4 {
                    println!(
                        "NOISE TEST FAIL: the field graph came back as {} nodes / {} edges",
                        n.graph.nodes.len(),
                        n.graph.edges.len()
                    );
                    std::process::exit(1);
                }
                // The Output node is what the graph evaluates to, and it
                // must be the one that survived the round trip.
                let out_id = n.graph.nodes.iter().find_map(|(id, k)| {
                    matches!(k, engine::field::FieldNode::Output).then_some(*id)
                });
                if out_id.is_none() || n.graph.output() != out_id {
                    println!(
                        "NOISE TEST FAIL: the graph does not resolve to its Output ({:?} vs {:?})",
                        n.graph.output(),
                        out_id
                    );
                    std::process::exit(1);
                }
                // And it has to be a real field, not a flat nothing: two
                // probes across it should read differently.
                // Phase 0: this stage is about what was authored, not
                // about it moving.
                let a = engine::field::eval_cpu(&n.graph, [0.1, 0.5], 0.0, 0.0, 1.0);
                let b = engine::field::eval_cpu(&n.graph, [0.9, 0.5], 0.0, 0.0, 1.0);
                if (a - b).abs() < 1e-4 {
                    println!("NOISE TEST FAIL: the authored field is flat ({a:.4})");
                    std::process::exit(1);
                }
                self.egraph.view = Some(nid);
                self.playback_dirty = true;
                next = 4;
            }
            4 if frames > 30 => {
                let Some(m) = self.probe_output_mean() else { return };
                if !(0.05..0.95).contains(&m) {
                    println!("NOISE TEST FAIL: the rendered field reads {m:.4}");
                    std::process::exit(1);
                }
                println!(
                    "NOISE TEST PASS: dived in, authored a driven-scale chain into the \
                     seeded Output, stepped out and rendered it (mean {m:.3})"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_noise = Some((next, if next == stage { frames } else { 0 }, nid));
    }
}
