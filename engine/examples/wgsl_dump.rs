//! Dump what codegen makes of a few Saturation graphs: the topology hash,
//! the fused WGSL and the packed uniforms. Run on two trees, diff.
use engine::graph::{BlurParams, DilateParams, GlowParams, Graph, NodeKind, SatDomain, SatLuma, SatParams};

fn main() {
    let cases = [
        (SatDomain::Linear, SatLuma::Working, true),
        (SatDomain::Log, SatLuma::Rec709, true),
        (SatDomain::Perceptual, SatLuma::Max, false),
        (SatDomain::Log, SatLuma::Average, false),
    ];
    for (domain, luma, masked) in cases {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::read(1, ""));
        g.nodes.insert(
            3,
            NodeKind::Saturation(SatParams {
                domain,
                luma,
                sat: 1.3,
                boost: 0.2,
                zones: [0.9, 1.1, 1.2],
                pivot: 0.2,
                width: 3.0,
                preserve: 0.5,
                compress: 0.25,
            }),
        );
        g.nodes.insert(4, NodeKind::Grade(Default::default()));
        g.edges.insert((3, 0), 1);
        if masked {
            g.edges.insert((3, 1), 2);
        }
        g.edges.insert((4, 0), 3);
        g.view = Some(4);
        let plan = engine::codegen::generate(&g);
        println!("== {domain:?} {luma:?} masked={masked}");
        println!("topo {:016x}", g.topology_hash());
        println!("slots {:?}", plan.layout.node_slot.get(&3));
        println!("{}", plan.main.wgsl);
        println!(
            "params {:?}",
            engine::codegen::pack_params(&g, &plan.layout, [64.0, 32.0])
        );
    }

    // Stage nodes: the fused side of a Blur / Dilate / Glow, masked and not,
    // plus each stage's own subtree shader.
    let stages: Vec<(&str, NodeKind, bool)> = vec![
        ("blur", NodeKind::Blur(BlurParams { size: 7.5, matte: false }), true),
        ("blur", NodeKind::Blur(BlurParams { size: 7.5, matte: false }), false),
        ("dilate", NodeKind::Dilate(DilateParams { amount: -2.5 }), false),
        ("glow", NodeKind::Glow(GlowParams { size: 90.0, ..Default::default() }), true),
        ("glow", NodeKind::Glow(GlowParams { size: 90.0, ..Default::default() }), false),
    ];
    for (name, kind, masked) in stages {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::read(1, ""));
        g.nodes.insert(3, kind);
        g.nodes.insert(4, NodeKind::Grade(Default::default()));
        g.edges.insert((3, 0), 1);
        if masked {
            g.edges.insert((3, 1), 2);
        }
        g.edges.insert((4, 0), 3);
        g.view = Some(4);
        let plan = engine::codegen::generate(&g);
        println!("== stage {name} masked={masked}");
        println!("topo {:016x}", g.topology_hash());
        println!("stages {}", plan.blurs.len());
        for (sid, _, sh, aux) in &plan.blurs {
            println!("-- stage {sid:?} subtree ({} aux)", aux.len());
            println!("{}", sh.wgsl);
        }
        println!("-- main");
        println!("{}", plan.main.wgsl);
        println!("params {:?}", engine::codegen::pack_params(&g, &plan.layout, [64.0, 32.0]));
    }
}
