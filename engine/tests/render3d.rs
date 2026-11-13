//! Render3D on real pixels: cards through a camera. The default camera
//! frames the default card exactly, so the render IS the card's picture;
//! a card behind another is hidden whatever order the scene lists them
//! in; the depth output is the distance to the card; and a card moving
//! across the shutter smears. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{CameraParams, CardParams, Graph, NodeKind, Render3dParams, SceneParams};
use engine::scene3d::Xform;
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 40;

fn source(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / W as f32, y as f32 / H as f32, 0.25, 1.0]
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP render3d test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, f: impl Fn(u32, u32) -> [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H).map(|i| f(i % W, i / W).map(f16::from_f32)).collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 8), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
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
    let mut out = Vec::new();
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
        for x in 0..W as usize {
            out.push([
                row[x * 4].to_f32(),
                row[x * 4 + 1].to_f32(),
                row[x * 4 + 2].to_f32(),
                row[x * 4 + 3].to_f32(),
            ]);
        }
    }
    drop(data);
    staging.unmap();
    out
}



fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, source);
    upload(&ctx, &mut comp, 1, |_, _| [1.0, 0.0, 0.0, 1.0]);
    upload(&ctx, &mut comp, 2, |_, _| [0.0, 1.0, 0.0, 1.0]);
    Some((ctx, comp))
}

/// Reads on slots 0.., one card per read, a Scene of the cards, a camera
/// and a Render3D (id 100) viewed at `out`.
fn scene(cards: &[(usize, CardParams)], cam: CameraParams, r: Render3dParams, out: u32) -> Graph {
    let mut g = Graph::default();
    let scene = 90;
    g.nodes.insert(scene, NodeKind::Scene3D(SceneParams { inputs: cards.len() }));
    for (i, (slot, card)) in cards.iter().enumerate() {
        let read = 10 + i as u64;
        let c = 20 + i as u64;
        g.nodes.insert(read, NodeKind::read(*slot, ""));
        g.nodes.insert(c, NodeKind::Card3D(card.clone()));
        g.edges.insert((c, 0), read);
        g.edges.insert((scene, i), c);
        g.formats.insert(read, (W, H));
        g.formats.insert(c, (W, H));
    }
    g.nodes.insert(91, NodeKind::Camera3D(cam));
    g.nodes.insert(100, NodeKind::Render3D(r));
    g.edges.insert((100, 0), 91);
    g.edges.insert((100, 1), scene);
    g.formats.insert(100, (W, H));
    g.view = Some(100);
    g.view_out = out;
    engine::scene3d::expand(&mut g, &|_, _| None);
    g
}

fn at(out: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    out[(y * W + x) as usize]
}

#[test]
fn the_default_camera_shows_the_default_card_as_it_is() {
    let Some((ctx, mut comp)) = setup() else { return };
    let g = scene(&[(0, CardParams::default())], CameraParams::default(), Default::default(), 0);
    let out = frame(&ctx, &mut comp, &g);
    let mut worst = 0.0f32;
    for y in 0..H {
        for x in 0..W {
            let (a, b) = (at(&out, x, y), source(x, y));
            for k in 0..4 {
                worst = worst.max((a[k] - b[k]).abs());
            }
        }
    }
    // Half-float output.
    assert!(worst < 2e-3, "worst {worst}");
}

#[test]
fn the_nearer_card_wins_whatever_the_scene_order() {
    let Some((ctx, mut comp)) = setup() else { return };
    let near = CardParams { xform: Xform { scale: [0.5; 3], ..Default::default() }, ..Default::default() };
    let far = CardParams {
        xform: Xform { translate: [0.0, 0.0, -1.0], scale: [2.0; 3], ..Default::default() },
        ..Default::default()
    };
    for order in [[(1, near.clone()), (2, far.clone())], [(2, far.clone()), (1, near.clone())]] {
        let out = frame(&ctx, &mut comp, &scene(&order, CameraParams::default(), Default::default(), 0));
        let c = at(&out, W / 2, H / 2);
        assert!(c[0] > 0.99 && c[1] < 0.01, "centre is the near (red) card: {c:?}");
        let e = at(&out, 2, 2);
        assert!(e[1] > 0.99 && e[0] < 0.01, "the corner is the far (green) card: {e:?}");
    }
}

#[test]
fn the_depth_output_is_the_distance_along_the_camera() {
    let Some((ctx, mut comp)) = setup() else { return };
    let cam = CameraParams::default();
    let d = cam.xform.translate[2];
    let g = scene(&[(0, CardParams::default())], cam.clone(), Default::default(), 1);
    let out = frame(&ctx, &mut comp, &g);
    let c = at(&out, W / 2, H / 2);
    assert!((c[0] - d).abs() < 2e-3, "depth {} want {d}", c[0]);
    // A tilted card is nearer on one side than the other.
    let tilt = CardParams { xform: Xform { rotate: [0.0, 30.0, 0.0], ..Default::default() }, ..Default::default() };
    let out = frame(&ctx, &mut comp, &scene(&[(0, tilt)], cam, Default::default(), 1));
    let (l, r) = (at(&out, W / 2 - 8, H / 2)[0], at(&out, W / 2 + 8, H / 2)[0]);
    // +30 about Y swings +X towards -Z: the right side goes back.
    assert!(l < d && r > d, "turned about Y the left side comes forward: {l} {r}");
}

#[test]
fn a_card_moving_across_the_shutter_smears() {
    let Some((ctx, mut comp)) = setup() else { return };
    let small = CardParams { xform: Xform { scale: [0.25; 3], ..Default::default() }, ..Default::default() };
    let r = Render3dParams { samples: 8, shutter: 1.0, ..Default::default() };
    let mut g = Graph::default();
    // Built by hand so the pose callback can move the card.
    g.nodes.insert(10, NodeKind::read(1, ""));
    g.nodes.insert(20, NodeKind::Card3D(small.clone()));
    g.edges.insert((20, 0), 10);
    g.formats.insert(10, (W, H));
    g.nodes.insert(91, NodeKind::Camera3D(CameraParams::default()));
    g.nodes.insert(100, NodeKind::Render3D(r));
    g.edges.insert((100, 0), 91);
    g.edges.insert((100, 1), 20);
    g.formats.insert(100, (W, H));
    g.view = Some(100);
    let moving = |id: u64, dt: f32| -> Option<NodeKind> {
        (id == 20).then(|| {
            let mut c = small.clone();
            c.xform.translate[0] = dt * 0.5;
            NodeKind::Card3D(c)
        })
    };
    engine::scene3d::expand(&mut g, &moving);
    let out = frame(&ctx, &mut comp, &g);
    let c = at(&out, W / 2, H / 2)[0];
    assert!(c > 0.3 && c < 0.95, "the centre is partly covered: {c}");
    let side = at(&out, W / 2 + 14, H / 2)[0];
    assert!(side > 0.05, "the smear reaches past the still card's edge: {side}");
}
