//! The Layer node: a stack of N images, each with its own transform, time
//! offset, opacity and blend op, composited bottom-up in ONE fused pass.
//!
//! What is worth pinning here is not the arithmetic of `over` — `blend_mix`
//! covers that — but the three things a stack adds and could silently break:
//! the compositing ORDER, the per-layer opacity, and the promise that
//! sliding a layer in time is a uniform write while adding one is not.
//! The GPU half self-skips without an adapter (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    BlendOp, Graph, LayerItem, LayerParams, NodeKind, TIME_KEY_ROOT,
};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP layer test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, v: [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|_| {
            [
                f16::from_f32(v[0]),
                f16::from_f32(v[1]),
                f16::from_f32(v[2]),
                f16::from_f32(v[3]),
            ]
        })
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> [f32; 4] {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    let mut warm = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut warm, &timer);
    ctx.queue.submit([warm.finish()]);
    comp.write_params(graph);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut encoder, &timer);
    let padded = (W * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: comp.output_texture().unwrap(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let halves: &[f16] = bytemuck::cast_slice(&data[..(W * 8) as usize]);
    let out =
        [halves[0].to_f32(), halves[1].to_f32(), halves[2].to_f32(), halves[3].to_f32()];
    drop(data);
    staging.unmap();
    out
}

fn item(opacity: f32, op: BlendOp) -> LayerItem {
    LayerItem { opacity, op, ..Default::default() }
}

/// `n` reads wired into one stack, top layer first.
fn stack(layers: Vec<LayerItem>) -> Graph {
    let mut g = Graph::default();
    let n = layers.len();
    for i in 0..n {
        g.nodes.insert(i as u64, NodeKind::read(i, ""));
        g.edges.insert((100, i), i as u64);
    }
    g.nodes.insert(100, NodeKind::Layer(LayerParams { layers, outputs: false }));
    g.view = Some(100);
    g
}

/// Layer 1 is the top: an opaque layer hides everything under it, and the
/// stack order is the ONLY thing that decides which one that is.
#[test]
fn the_top_layer_is_the_one_you_see() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [1.0, 0.0, 0.0, 1.0]);
    upload(&ctx, &mut comp, 1, [0.0, 1.0, 0.0, 1.0]);
    upload(&ctx, &mut comp, 2, [0.0, 0.0, 1.0, 1.0]);

    let g = stack(vec![item(1.0, BlendOp::Over); 3]);
    let got = render(&ctx, &mut comp, &g);
    assert!(got[0] > 0.9 && got[1] < 0.1, "top layer (red) should win, got {got:?}");

    // Swap the wires: the same three plates, restacked, show the middle one.
    let mut g2 = g.clone();
    g2.edges.insert((100, 0), 1);
    g2.edges.insert((100, 1), 0);
    let got = render(&ctx, &mut comp, &g2);
    assert!(got[1] > 0.9 && got[0] < 0.1, "restacked, green should win, got {got:?}");
}

/// Opacity is premultiplied: half a red layer over an opaque blue one is
/// half of each, and alpha stays 1 because the bottom layer is opaque.
#[test]
fn opacity_fades_a_layer_into_the_stack() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [1.0, 0.0, 0.0, 1.0]);
    upload(&ctx, &mut comp, 1, [0.0, 0.0, 1.0, 1.0]);
    for (o, want) in [(0.0f32, [0.0f32, 0.0, 1.0]), (0.5, [0.5, 0.0, 0.5]), (1.0, [1.0, 0.0, 0.0])]
    {
        let g = stack(vec![item(o, BlendOp::Over), item(1.0, BlendOp::Over)]);
        let got = render(&ctx, &mut comp, &g);
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < 3e-3,
                "opacity {o}, channel {c}: got {} want {}",
                got[c],
                want[c]
            );
        }
        assert!((got[3] - 1.0).abs() < 3e-3, "opaque base, alpha should stay 1");
    }
}

/// The TRIM gate is the app's per-tick stamp, folded into the same uniform
/// as opacity — a layer outside its window contributes nothing at all.
#[test]
fn the_trim_gate_removes_a_layer_entirely() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [1.0, 0.0, 0.0, 1.0]);
    upload(&ctx, &mut comp, 1, [0.0, 0.0, 1.0, 1.0]);
    let mut top = item(1.0, BlendOp::Over);
    top.gate = 0.0;
    let g = stack(vec![top, item(1.0, BlendOp::Over)]);
    let got = render(&ctx, &mut comp, &g);
    assert!(got[2] > 0.9 && got[0] < 0.1, "gated-off layer still showed: {got:?}");
}

/// Mute takes a layer out of the composite; a solo anywhere in the stack
/// takes out every layer that is not soloed; mute wins over solo.
#[test]
fn mute_and_solo_pick_what_composites() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [1.0, 0.0, 0.0, 1.0]);
    upload(&ctx, &mut comp, 1, [0.0, 1.0, 0.0, 1.0]);
    upload(&ctx, &mut comp, 2, [0.0, 0.0, 1.0, 1.0]);
    let base = || vec![item(1.0, BlendOp::Over); 3];

    let mut l = base();
    l[0].mute = true;
    let got = render(&ctx, &mut comp, &stack(l));
    assert!(got[1] > 0.9 && got[0] < 0.1, "muted top: green should show, got {got:?}");

    let mut l = base();
    l[2].solo = true;
    let got = render(&ctx, &mut comp, &stack(l));
    assert!(got[2] > 0.9 && got[0] < 0.1 && got[1] < 0.1, "solo bottom: blue alone, got {got:?}");

    let mut l = base();
    l[1].solo = true;
    l[1].mute = true;
    l[2].solo = true;
    let got = render(&ctx, &mut comp, &stack(l));
    assert!(got[2] > 0.9 && got[1] < 0.1, "muted solo stays out, got {got:?}");
}

/// The bottom layer is the BASE: it has nothing to land on, so its op is
/// ignored. Without this a bottom layer set to multiply multiplies itself
/// into black and the comp goes dark for no visible reason.
#[test]
fn the_bottom_layers_op_is_ignored() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [0.0, 0.0, 0.0, 0.0]);
    upload(&ctx, &mut comp, 1, [0.25, 0.5, 0.75, 1.0]);
    let g = stack(vec![item(1.0, BlendOp::Over), item(1.0, BlendOp::Multiply)]);
    let got = render(&ctx, &mut comp, &g);
    for (c, want) in [0.25f32, 0.5, 0.75].into_iter().enumerate() {
        assert!((got[c] - want) < 3e-3, "base should pass through, got {got:?}");
    }
}

/// The invariant the whole node design rests on: sliding a layer in time,
/// fading it or moving it is a UNIFORM write, while adding one or changing
/// what it does to the picture is a recompile.
#[test]
fn only_structure_recompiles() {
    let base = stack(vec![item(1.0, BlendOp::Over); 2]);
    let h = base.topology_hash();

    let mut slid = base.clone();
    if let Some(NodeKind::Layer(p)) = slid.nodes.get_mut(&100) {
        p.layers[0].start = 12;
        p.layers[0].opacity = 0.3;
        p.layers[0].trim = Some((2, 9));
        p.layers[0].mute = true;
        p.layers[1].solo = true;
    }
    assert_eq!(slid.topology_hash(), h, "a time/opacity edit recompiled");

    let mut op = base.clone();
    if let Some(NodeKind::Layer(p)) = op.nodes.get_mut(&100) {
        p.layers[0].op = BlendOp::Screen;
    }
    assert_ne!(op.topology_hash(), h, "a different blend formula is a different shader");

    let grown = stack(vec![item(1.0, BlendOp::Over); 3]);
    assert_ne!(grown.topology_hash(), h, "a new port is a new shader");
}

/// Each layer is its own branch of time, so ONE plate wired into three
/// layers at three starts decodes three frames and all three are live at
/// once. That is what makes a stack an edit rather than three copies of the
/// same frame.
#[test]
fn every_layer_gets_its_own_reader_instance() {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(
        100,
        NodeKind::Layer(LayerParams {
            outputs: false,
            layers: vec![
                LayerItem { start: 0, ..Default::default() },
                LayerItem { start: 5, ..Default::default() },
                LayerItem { start: -3, ..Default::default() },
            ],
        }),
    );
    for i in 0..3 {
        g.edges.insert((100, i), 1);
    }
    g.view = Some(100);

    let inst = g.reader_instances();
    assert_eq!(inst.len(), 3, "one instance per layer, got {inst:?}");
    let mut keys: Vec<_> = inst.iter().map(|i| i.key).collect();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), 3, "layers must not share a time path");

    // source = comp − start: the layer that STARTS at 5 shows its own frame
    // 0 there, and the one that starts at −3 is already 3 frames in.
    let at = |start: i64, tick: u64| {
        let want = engine::graph::input_time_key(
            TIME_KEY_ROOT,
            100,
            match start {
                0 => 0,
                5 => 1,
                _ => 2,
            },
        );
        inst.iter().find(|i| i.key == want).unwrap().map.apply(tick)
    };
    assert_eq!(at(0, 10), 10);
    assert_eq!(at(5, 10), 5);
    assert_eq!(at(-3, 10), 13);

    // And the keys are STRUCTURAL: re-timing a layer must not move them, or
    // dragging a bar past zero would recompile.
    let h = g.topology_hash();
    if let Some(NodeKind::Layer(p)) = g.nodes.get_mut(&100) {
        p.layers[1].start = 40;
    }
    assert_eq!(g.reader_instances().len(), 3);
    assert_eq!(g.topology_hash(), h, "re-timing a layer moved a reader slot");
}

/// Stretching a layer's bar is a RETIME: the same source frames spread over
/// more comp frames. It rides the same per-layer TimeMap as the start, so it
/// is a uniform-only edit like everything else on the timeline.
#[test]
fn a_stretched_layer_plays_slower() {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(
        100,
        NodeKind::Layer(LayerParams {
            outputs: false,
            layers: vec![
                // Half speed from frame 0: source 0,0,1,1,2,2...
                LayerItem { speed: 0.5, ..Default::default() },
                // Double speed, starting late.
                LayerItem { start: 10, speed: 2.0, ..Default::default() },
            ],
        }),
    );
    g.edges.insert((100, 0), 1);
    g.edges.insert((100, 1), 1);
    g.view = Some(100);

    let h = g.topology_hash();
    let inst = g.reader_instances();
    let at = |layer: usize, tick: u64| {
        let k = engine::graph::input_time_key(TIME_KEY_ROOT, 100, layer);
        inst.iter().find(|i| i.key == k).unwrap().map.apply(tick)
    };
    assert_eq!(at(0, 0), 0);
    assert_eq!(at(0, 10), 5, "half speed reaches source 5 at comp 10");
    assert_eq!(at(1, 10), 0, "a late layer starts at its own frame 0");
    assert_eq!(at(1, 16), 12, "double speed is 12 source frames in by comp 16");

    // And it is a uniform edit: no recompile, no reader slot moved.
    if let Some(NodeKind::Layer(p)) = g.nodes.get_mut(&100) {
        p.layers[0].speed = 0.25;
    }
    assert_eq!(g.topology_hash(), h, "a retime recompiled");
    assert_eq!(g.reader_instances().len(), 2);

    // A rate of zero would freeze the layer AND divide by zero working out
    // its bar length, so it is guarded rather than trusted.
    let frozen = LayerItem { speed: 0.0, ..Default::default() };
    assert!(frozen.rate() != 0.0);
    assert!(frozen.stretched(24) > 0);
}

/// A stack with no layers is a legal (if useless) node: it must render as
/// transparent rather than reaching for a uniform slot it does not own.
#[test]
fn an_empty_stack_is_transparent() {
    let mut g = Graph::default();
    g.nodes.insert(100, NodeKind::Layer(LayerParams { layers: Vec::new(), outputs: false }));
    g.view = Some(100);
    let plan = engine::codegen::generate(&g);
    assert!(plan.main.wgsl.contains("@compute"));
    assert!(plan.main.sources.is_empty(), "an empty stack samples nothing");
}

/// A stack has to survive a setup file: everything authored comes back, and
/// the transient trim GATE comes back as 1 rather than as whatever tick the
/// file happened to be saved on.
#[test]
#[cfg(feature = "serde")]
fn a_stack_round_trips_through_a_setup() {
    let mut p = LayerParams {
        outputs: false,
        layers: vec![
            LayerItem {
                name: "sky".into(),
                start: 12,
                opacity: 0.4,
                speed: 0.5,
                op: BlendOp::Screen,
                trim: Some((3, 40)),
                legacy_xform: None,
                gate: 0.0,
                extend: None,
                mute: true,
                solo: true,
            },
            LayerItem::default(),
        ],
    };
    let text = serde_json::to_string(&NodeKind::Layer(p.clone())).unwrap();
    let back: NodeKind = serde_json::from_str(&text).unwrap();
    // The gate is the app's per-tick stamp, never authored data.
    p.layers[0].gate = 1.0;
    assert_eq!(back, NodeKind::Layer(p));

    // And a stack written before trims existed still loads (they are the
    // kind of field that gets added, and an old setup must not fail).
    let old = r#"{"Layer":{"layers":[{"xform":{"translate":[0.0,0.0],"rotate":0.0,
        "scale":[1.0,1.0],"center":[0.5,0.5]},"opacity":1.0}]}}"#;
    // (that one predates `speed` as well as `trim` — both must default)
    let NodeKind::Layer(loaded) = serde_json::from_str::<NodeKind>(old).unwrap() else {
        panic!("not a stack")
    };
    assert_eq!(loaded.layers.len(), 1);
    assert_eq!(loaded.layers[0].trim, None);
    assert_eq!(loaded.layers[0].start, 0);
    assert_eq!(loaded.layers[0].gate, 1.0);
    assert_eq!(loaded.layers[0].speed, 1.0);
    // Its transform is kept, as legacy, for the app to lift into a
    // Transform node — and a lifted stack writes no `xform` at all.
    assert!(loaded.layers[0].legacy_xform.is_some(), "an old layer transform was dropped");
    assert!(!text.contains("xform"), "a stack with no legacy transform wrote one");
}

/// With layer outputs on, output 1 + i is layer i ALONE: none of the other
/// layers, none of its own opacity or blend op — but its trim still gates
/// it (a time operation, not a comp one).
#[test]
fn a_layer_output_is_that_layer_alone() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [1.0, 0.0, 0.0, 1.0]);
    upload(&ctx, &mut comp, 1, [0.0, 1.0, 0.0, 1.0]);

    let mut g = stack(vec![item(0.3, BlendOp::Screen), item(1.0, BlendOp::Over)]);
    if let Some(NodeKind::Layer(p)) = g.nodes.get_mut(&100) {
        p.outputs = true;
    }
    let view = |g: &Graph, out: u32| {
        let mut g = g.clone();
        g.view_out = out;
        engine::layerouts::expand(&mut g);
        g
    };
    let c = render(&ctx, &mut comp, &view(&g, 0));
    assert!(c[0] > 0.2 && c[1] > 0.9, "output 0 is the comp, got {c:?}");
    let top = render(&ctx, &mut comp, &view(&g, 1));
    assert!((top[0] - 1.0).abs() < 1e-3 && top[1] < 1e-3, "layer 1 alone, opaque: {top:?}");
    let bottom = render(&ctx, &mut comp, &view(&g, 2));
    assert!(bottom[0] < 1e-3 && (bottom[1] - 1.0).abs() < 1e-3, "layer 2 alone: {bottom:?}");

    // Outside its trim the layer's own output is empty, as it is in the comp.
    if let Some(NodeKind::Layer(p)) = g.nodes.get_mut(&100) {
        p.layers[0].gate = 0.0;
    }
    let gated = render(&ctx, &mut comp, &view(&g, 1));
    assert!(gated[3] < 1e-3, "a trimmed-out layer shows nothing: {gated:?}");
}
