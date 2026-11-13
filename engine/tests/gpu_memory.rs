//! GPU memory that must be RELEASED, not just allocated. Two leaks crashed a
//! 6K comp on a 24 GB card ("Out of Memory" after half an hour, sooner with
//! a Defocus in the tree): reader textures for retired slots, and the
//! scratch of a Defocus that had left the graph. Both kept canvas-sized
//! textures alive for the rest of the session. These pin the release.
//!
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{DefocusParams, Graph, NodeKind};
use engine::timing::GpuTimer;

const W: u32 = 128;
const H: u32 = 128;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP gpu_memory test: no adapter ({e})");
            None
        }
    }
}

fn encode(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut enc, &timer);
    ctx.queue.submit([enc.finish()]);
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
}

fn plate() -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.view = Some(1);
    g
}

fn defocused() -> Graph {
    let mut g = plate();
    g.nodes.insert(2, NodeKind::Defocus(DefocusParams::default()));
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g
}

#[test]
fn a_removed_defocus_gives_its_scratch_back() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    comp.ensure_output(W, H);

    encode(&ctx, &mut comp, &defocused());
    assert_eq!(comp.stage_scratch_count(), 1, "a Defocus allocates its chain");

    encode(&ctx, &mut comp, &plate());
    assert_eq!(comp.stage_scratch_count(), 0, "the Defocus left; its textures must too");

    // And back: re-adding allocates afresh (and still renders).
    encode(&ctx, &mut comp, &defocused());
    assert_eq!(comp.stage_scratch_count(), 1);
}

#[test]
fn retired_reader_slots_release_their_textures() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    let info = ImageInfo { width: W, height: H };
    for slot in 0..5 {
        assert!(comp.ensure_reader(slot, info));
    }
    let freed = comp.retain_readers(&[1, 3]);
    assert_eq!(freed, 3 * W as u64 * H as u64 * 8, "slots 0, 2 and 4 freed");
    for slot in 0..5 {
        assert_eq!(comp.reader_texture(slot).is_some(), slot == 1 || slot == 3, "slot {slot}");
    }
    // Nothing more to free the second time round.
    assert_eq!(comp.retain_readers(&[1, 3]), 0);
}

#[test]
fn a_resized_reader_replaces_its_texture() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    assert!(comp.ensure_reader(0, ImageInfo { width: W, height: H }));
    assert!(comp.ensure_reader(0, ImageInfo { width: W / 2, height: H / 2 }));
    let t = comp.reader_texture(0).expect("texture");
    assert_eq!((t.info.width, t.info.height), (W / 2, H / 2));
}
