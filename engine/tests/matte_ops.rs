//! Pointwise ops on a MATTE: an Invert, Gamma or Mult sitting in a matte
//! pipe works on the coverage and hands back a matte — the coverage in all
//! four channels, the way every matte source writes it — and a mask port
//! downstream reads the result. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    GammaParams, GradeParams, Graph, MatteChannel, MultParams, NodeKind, ToMatteParams,
};
use engine::plugins::{self, ShaderNodeDef};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 32;
const H: u32 = 1;

/// A coverage ramp in red, 0..1, over a grey picture.
fn source(x: u32) -> [f32; 4] {
    let t = x as f32 / (W - 1) as f32;
    [t, 0.5, 0.5, 1.0]
}

/// The SHIPPED Invert, read off its manifest (the engine's own tests do not
/// build the manifest parser), so the flag that lets it take a matte is the
/// one users get.
fn register_invert() {
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../plugins/invert/plugin.json"),
    )
    .unwrap();
    let m: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(m["mattes"], true, "the Invert plugin must declare that it takes mattes");
    let def = ShaderNodeDef::new(
        "invert".into(),
        "Invert".into(),
        [0; 3],
        1,
        1,
        vec![plugins::ParamSpec {
            name: "amount".into(),
            label: "amount".into(),
            default: 1.0,
            min: 0.0,
            max: 1.0,
        }],
        m["expr"].as_str().unwrap().into(),
        String::new(),
    )
    .with_mattes(true);
    plugins::register(def);
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = match engine::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP matte ops test: no adapter ({e})");
            return None;
        }
    };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W).map(|x| source(x).map(f16::from_f32)).collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 8), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    Some((ctx, comp))
}

fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
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
    let row: &[f16] = bytemuck::cast_slice(&data[..(W * 8) as usize]);
    let out = (0..W as usize)
        .map(|x| std::array::from_fn(|c| row[x * 4 + c].to_f32()))
        .collect();
    drop(data);
    staging.unmap();
    out
}

/// Read -> ToMatte (red) -> `op` in the matte pipe, viewed.
fn matte_chain(mut op: NodeKind) -> Graph {
    assert!(op.follows_pipe(0), "{op:?} should follow its pipe");
    op.set_matte_mode(true);
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(
        1,
        NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::R, ..Default::default() }),
    );
    g.nodes.insert(2, op);
    g.edges.insert((1, 0), 0);
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.004 + 0.004 * b.abs()
}

/// Every channel of every pixel is `f(coverage)`: a matte stays a matte.
fn assert_matte(name: &str, out: &[[f32; 4]], f: impl Fn(f32) -> f32) {
    for (x, px) in out.iter().enumerate() {
        let want = f(source(x as u32)[0]);
        for (c, v) in px.iter().enumerate() {
            assert!(close(*v, want), "{name}: x={x} ch{c} is {v}, the matte says {want}");
        }
    }
}

#[test]
fn pointwise_ops_on_a_matte_give_a_matte() {
    let Some((ctx, mut comp)) = setup() else { return };
    register_invert();
    let invert = NodeKind::Custom { type_id: "invert".into(), params: Default::default() };
    assert_matte("invert", &frame(&ctx, &mut comp, &matte_chain(invert)), |c| 1.0 - c);
    let gamma = NodeKind::Gamma(GammaParams { master: 2.0, ..Default::default() });
    assert_matte("gamma", &frame(&ctx, &mut comp, &matte_chain(gamma)), |c| c.sqrt());
    let mult = NodeKind::Mult(MultParams { value: [0.5; 3], ..Default::default() });
    assert_matte("mult", &frame(&ctx, &mut comp, &matte_chain(mult)), |c| c * 0.5);
    let grade = NodeKind::Grade(GradeParams { gain: [0.25; 3], ..Default::default() });
    assert_matte("grade", &frame(&ctx, &mut comp, &matte_chain(grade)), |c| c * 0.25);
}

/// The whole point of an Invert on a matte: a Grade's mask port takes it,
/// and the grade lands where the matte is NOT.
#[test]
fn an_inverted_matte_masks_a_grade() {
    let Some((ctx, mut comp)) = setup() else { return };
    register_invert();
    let mut g = matte_chain(NodeKind::Custom { type_id: "invert".into(), params: Default::default() });
    g.nodes.insert(3, NodeKind::Grade(GradeParams { gain: [2.0; 3], ..Default::default() }));
    g.edges.insert((3, 0), 0);
    g.edges.insert((3, 1), 2);
    g.view = Some(3);
    let out = frame(&ctx, &mut comp, &g);
    for (x, px) in out.iter().enumerate() {
        let s = source(x as u32);
        let m = 1.0 - s[0];
        let want = s[1] * (1.0 + m); // mix(s, 2s, m) on the grey channel
        assert!(close(px[1], want), "x={x}: graded {} wants {want}", px[1]);
    }
}

/// Matte mode is topology (it adds the splat), but only when it is on: an
/// image-pipe op hashes exactly as it always did.
#[test]
fn matte_mode_is_topology_and_only_when_on() {
    let img = matte_chain(NodeKind::Gamma(GammaParams::default()));
    let mut plain = img.clone();
    plain.nodes.get_mut(&2).unwrap().set_matte_mode(false);
    assert_ne!(img.topology_hash(), plain.topology_hash());
    assert_eq!(
        NodeKind::Gamma(GammaParams::default()),
        *plain.nodes.get(&2).unwrap(),
        "out of the matte pipe a Gamma is the Gamma it always was"
    );
}
