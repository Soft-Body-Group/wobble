//! Proxy mode stores every pixel param at FULL resolution and scales it once,
//! at pack time, by `Graph::proxy_scale`. A param that skips that step renders
//! at twice its intended offset under proxy, and a manipulator that skips the
//! matching step on the app side moves the image half as far as the pointer.
//!
//! With identity rotate and scale the packed translation term is exactly
//! `-translate` whatever the canvas size (the pivot terms cancel), so under
//! proxy ½ it must be exactly half — no tolerance needed, the factor is a
//! power of two.

use engine::codegen::{generate, pack_params};
use engine::graph::{Graph, NodeKind, TransformParams};

const FULL: [f32; 2] = [2048.0, 1024.0];

fn packed_tx(g: &Graph, node: u64, offset: usize, out: [f32; 2]) -> f32 {
    let plan = generate(g);
    let v = pack_params(g, &plan.layout, out);
    let base = plan.layout.node_slot[&node] * 4;
    v[base + offset]
}

fn moved(x: f32, y: f32) -> TransformParams {
    TransformParams { translate: [x, y], ..Default::default() }
}

#[test]
fn transform_translate_halves_under_proxy() {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Transform(moved(100.0, 40.0)));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);

    let full = packed_tx(&g, 1, 2, FULL);
    g.proxy = 0.5;
    let half = packed_tx(&g, 1, 2, [FULL[0] / 2.0, FULL[1] / 2.0]);
    assert_eq!(full, -100.0, "full-res translation term");
    assert_eq!(half, -50.0, "proxy must pack half the pixels");
}
