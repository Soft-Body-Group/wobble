//! The 3D cards' Blender sync against a REAL Blender: launch it on a known
//! scene, ask for the scene's cameras and planes over the live session,
//! and render cards placed by what came back. The plane is two units wide
//! at the origin under a camera five units up, so its card must sit in the
//! middle of the frame at the size the lens says, and the corners see past
//! it.
//!
//! Ignored: needs Blender (`WOBBLE_BLENDER` or /Applications) and a
//! display, and opens a Blender window for a few seconds.
//!
//!     cargo test -p engine --features serde --test blender_scene3d -- --ignored --nocapture

use engine::blender::{self, live};
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{CameraParams, CardParams, Graph, NodeKind, Render3dParams};
use engine::scene3d::Source;
use engine::timing::GpuTimer;
use half::f16;
use std::time::{Duration, Instant};

const W: u32 = 64;
const H: u32 = 40;

/// The plane, the camera straight above it, and a still range.
const MAKE_SCENE: &str = r#"
import bpy, sys
out = sys.argv[sys.argv.index("--") + 1]
sc = bpy.context.scene
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.mesh.primitive_plane_add(size=2)
bpy.ops.object.camera_add(location=(0, 0, 5)); sc.camera = bpy.context.object
sc.camera.data.lens = 50; sc.camera.data.sensor_width = 36
sc.render.resolution_x, sc.render.resolution_y = 1920, 1080
sc.frame_start, sc.frame_end = 1, 8
bpy.ops.wm.save_as_mainfile(filepath=out)
"#;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP blender scene3d test: no adapter ({e})");
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




#[test]
#[ignore]
fn a_real_blender_places_cards_through_its_camera() {
    let Some(ctx) = ctx() else { return };
    let exe = blender::executable();
    let dir = std::env::temp_dir().join(format!("wobble-blender-scene3d-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let make = dir.join("make_scene.py");
    std::fs::write(&make, MAKE_SCENE).unwrap();
    let blend = dir.join("probe.blend");
    let out = std::process::Command::new(&exe)
        .args(["-b", "--factory-startup", "-P"])
        .arg(&make)
        .arg("--")
        .arg(&blend)
        .output()
        .expect("running blender headless");
    assert!(blend.exists(), "scene not written:\n{}", String::from_utf8_lossy(&out.stdout));

    let script = blender::install_script(&dir).unwrap();
    let port = live::free_port().unwrap();
    let s = live::Session::launch(&exe, Some(&blend), &script, port, &dir).unwrap();
    let t0 = Instant::now();
    let mut hello = false;
    let mut file = None;
    let want = dir.join("scene3d.json");
    while file.is_none() {
        assert!(t0.elapsed() < Duration::from_secs(90), "no scene3d answer");
        assert!(!s.closed(), "session closed");
        for ev in s.poll() {
            match ev {
                live::Event::Hello { .. } => {
                    hello = true;
                    s.scene3d(&want.to_string_lossy());
                }
                live::Event::Scene3d { path } => file = Some(path),
                live::Event::Error(e) => eprintln!("  ERROR {e}"),
                _ => {}
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(hello);
    let path = file.unwrap();
    eprintln!("scene3d in {:?}: {path}", t0.elapsed());
    s.quit();
    std::thread::sleep(Duration::from_millis(300));
    s.kill();

    // What came back reads in our convention.
    let f = engine::scene3d::import::load(&path).unwrap();
    let names: Vec<&str> = f.tracks.iter().map(|t| t.name.as_str()).collect();
    eprintln!("objects {names:?}");
    let (cam, lens) = f.track("", engine::scene3d::import::TrackKind::Camera).unwrap().sample(1.0).unwrap();
    assert!((cam.translate[1] - 5.0).abs() < 1e-3, "Blender z=5 is our y=5: {cam:?}");
    assert!((lens.unwrap().focal - 50.0).abs() < 1e-3);

    // Through the nodes: camera and card both from the Blender node.
    engine::scene3d::set_blender_scene(7, path.clone(), 0.0);
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, |_, _| [0.25, 0.5, 0.75, 1.0]);
    let mut g = Graph::default();
    g.nodes.insert(10, NodeKind::read(0, ""));
    g.formats.insert(10, (W, W));
    let mut cam_node = NodeKind::Camera3D(CameraParams {
        source: Source::Blender { node: 7, object: String::new() },
        ..Default::default()
    });
    let mut card_node = NodeKind::Card3D(CardParams {
        source: Source::Blender { node: 7, object: "Plane".into() },
        ..Default::default()
    });
    engine::scene3d::apply_file_source(&mut cam_node, 1.0).unwrap();
    engine::scene3d::apply_file_source(&mut card_node, 1.0).unwrap();
    g.nodes.insert(20, card_node);
    g.edges.insert((20, 0), 10);
    g.nodes.insert(91, cam_node);
    g.nodes.insert(100, NodeKind::Render3D(Render3dParams::default()));
    g.edges.insert((100, 0), 91);
    g.edges.insert((100, 1), 20);
    g.formats.insert(100, (W, H));
    g.view = Some(100);
    engine::scene3d::expand(&mut g, &|_, _| None);
    let px = frame(&ctx, &mut comp, &g);
    let at = |x: u32, y: u32| px[(y * W + x) as usize];
    let c = at(W / 2, H / 2);
    assert!((c[0] - 0.25).abs() < 0.01 && (c[2] - 0.75).abs() < 0.01, "the plane is in the middle: {c:?}");
    assert!(at(1, 1)[3] < 0.01, "the corner sees past it: {:?}", at(1, 1));
    // A 2-unit plane 5 units from a 50mm on 36mm: 2/5*50/36 of the width.
    let row: Vec<f32> = (0..W).map(|x| at(x, H / 2)[3]).collect();
    let covered = row.iter().filter(|a| **a > 0.5).count() as f32;
    let want = 2.0 / 5.0 * 50.0 / 36.0 * W as f32;
    assert!((covered - want).abs() <= 1.5, "card spans {covered}px, want {want}");
}
