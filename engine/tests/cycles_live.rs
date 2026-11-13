//! Live Cycles straight into wobble, against a REAL wobble-fork Blender:
//! the fork announces itself, wobble hands it a shared Vulkan buffer, a
//! final (F12) Cycles render on the GPU (CUDA on Windows, Metal on macOS)
//! writes its progressive
//! updates into that buffer, and the pixels read back off wobble's own
//! device are the scene: an emission plane of exactly 0.5.
//!
//! Ignored: needs a fork build (`WOBBLE_BLENDER`), an NVIDIA GPU (Windows)
//! or Apple silicon (macOS) and a display, and opens a Blender window.
//!
//!     WOBBLE_BLENDER=...\blender.exe cargo test -p engine --test cycles_live -- --ignored --nocapture
//!     WOBBLE_BLENDER=~/src/devl/build_darwin/bin/Blender.app/Contents/MacOS/Blender cargo test -p engine --test cycles_live -- --ignored --nocapture

#![cfg(any(windows, target_os = "macos"))]

use engine::blender::{self, cycles::CyclesTarget, live, Quality};
use std::time::{Duration, Instant};

const W: u32 = 256;
const H: u32 = 128;

const MAKE_SCENE: &str = r#"
import bpy, sys
out = sys.argv[sys.argv.index("--") + 1]
sc = bpy.context.scene
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.mesh.primitive_plane_add(size=2)
m = bpy.data.materials.new("e"); m.use_nodes = True
nt = m.node_tree; nt.nodes.clear()
em = nt.nodes.new("ShaderNodeEmission"); em.inputs[1].default_value = 0.5
o = nt.nodes.new("ShaderNodeOutputMaterial"); nt.links.new(em.outputs[0], o.inputs[0])
bpy.context.object.data.materials.append(m)
bpy.ops.object.camera_add(location=(0, 0, 5)); sc.camera = bpy.context.object
sc.render.engine = 'CYCLES'
sc.cycles.samples = 256
sc.frame_start, sc.frame_end = 1, 8
bpy.ops.wm.save_as_mainfile(filepath=out)
"#;

/// Half4 pixel (x, y) of the shared buffer, read back through wobble's device.
/// Rows are packed at `w * 8`, or padded where the stamp says so (macOS).
fn read_px(ctx: &engine::GpuContext, t: &CyclesTarget, w: u32, x: u32, y: u32) -> [f32; 4] {
    let stride = if t.stamp().version == blender::cycles::STAMP_PADDED {
        blender::cycles::padded_stride(w) as u64
    } else {
        w as u64 * 8
    };
    let off = y as u64 * stride + x as u64 * 8;
    let rb = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cycles-readback"),
        size: 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&t.shared, off, &rb, 0, 8);
    ctx.queue.submit([enc.finish()]);
    rb.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = rb.slice(..).get_mapped_range().unwrap();
    let h: &[half::f16] = bytemuck::cast_slice(&data);
    [h[0].to_f32(), h[1].to_f32(), h[2].to_f32(), h[3].to_f32()]
}

#[test]
#[ignore]
fn a_fork_renders_cycles_into_wobbles_buffer() {
    let exe = blender::executable();
    let ctx = engine::GpuContext::new().unwrap();
    let dir = std::env::temp_dir().join(format!("wobble-cycles-live-{}", std::process::id()));
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

    // The fork says so after hello, with its pid.
    let t0 = Instant::now();
    let pid = loop {
        assert!(t0.elapsed() < Duration::from_secs(90), "no cycles_fork: is WOBBLE_BLENDER a fork build?");
        assert!(!s.closed(), "session closed");
        let hit = s.poll().into_iter().find_map(|ev| match ev {
            live::Event::CyclesFork { pid } => Some(pid),
            live::Event::Log(l) => {
                eprintln!("  {l}");
                None
            }
            _ => None,
        });
        if let Some(p) = hit {
            break p;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    eprintln!("fork pid {pid}");

    // Room for a live viewport render too (as big as the region).
    let mut target = CyclesTarget::new(&ctx.device, 4096, 2304, "test").unwrap();
    let handle = target.handle_for(pid).unwrap();
    s.cycles(handle, target.size, &target.uuid_hex(), &target.stamp_name);

    let p = blender::BlenderParams {
        engine: blender::BlenderEngine::Cycles,
        quality: Quality::Final,
        transparent: false,
        ..Default::default()
    };
    s.spec(&p.spec_line(W, H));
    s.frame(3);

    // Progressive updates land in OUR buffer while the render runs.
    let t0 = Instant::now();
    let mut updates = Vec::new();
    let mut bound = false;
    while t0.elapsed() < Duration::from_secs(120) {
        for ev in s.poll() {
            match ev {
                live::Event::CyclesBound { ok, msg } => {
                    assert!(ok, "the fork refused the target: {msg}");
                    bound = true;
                }
                live::Event::Log(l) => eprintln!("  {l}"),
                live::Event::Error(e) => eprintln!("  error: {e}"),
                _ => {}
            }
        }
        if let Some(u) = target.poll() {
            let c = read_px(&ctx, &target, u.w, u.w / 2, u.h / 2);
            eprintln!("  update {} at {:?}: {}x{} centre {c:?}", u.updates, t0.elapsed(), u.w, u.h);
            updates.push((u, c));
            if (c[0] - 0.5).abs() < 0.02 {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(bound, "the fork never answered the target");
    assert!(!updates.is_empty(), "no Cycles update reached wobble's buffer");
    let (u, c) = updates.last().copied().unwrap();
    assert_eq!((u.w, u.h), (W, H));
    assert!((c[0] - 0.5).abs() < 0.02 && (c[1] - 0.5).abs() < 0.02, "centre {c:?}");
    eprintln!("final: {} progressive updates straight into wobble", updates.len());

    // 2. LIVE: the viewport's own Cycles session, camera frame only,
    // rendering continuously into the same buffer.
    let live = blender::BlenderParams { quality: Quality::Viewport, shading: blender::Shading::Rendered, ..p };
    s.spec(&live.spec_line(W, H));
    s.frame(4);
    let t0 = Instant::now();
    let mut n = 0u32;
    let mut first = None;
    let mut last = None;
    while t0.elapsed() < Duration::from_secs(20) {
        for ev in s.poll() {
            if let live::Event::Log(l) | live::Event::Error(l) = ev {
                eprintln!("  {l}");
            }
        }
        if let Some(u) = target.poll() {
            n += 1;
            first.get_or_insert(t0.elapsed());
            last = Some((u, read_px(&ctx, &target, u.w, u.w / 2, u.h / 2)));
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let (lu, lc) = last.expect("no live viewport update reached wobble");
    eprintln!("live: {n} updates in 20s, first after {:?}, last {}x{} centre {lc:?}", first.unwrap(), lu.w, lu.h);
    assert!((lc[0] - 0.5).abs() < 0.05, "live centre {lc:?}");

    // 3. A new frame restarts it: updates keep coming.
    s.frame(5);
    let t0 = Instant::now();
    let mut m = 0u32;
    while t0.elapsed() < Duration::from_secs(5) {
        let _ = s.poll();
        if target.poll().is_some() {
            m += 1;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    eprintln!("after a frame change: {m} updates in 5s");
    assert!(m > 0, "a frame change did not restart the live render");
    s.quit();
    // The dirty scene stops `quit` at the save prompt: end it outright.
    std::thread::sleep(Duration::from_millis(500));
    let _ = std::process::Command::new("taskkill").args(["/PID", &pid.to_string(), "/F"]).output();
}

/// Stands in for the user's hand and eye in Blender: a timer that sets the
/// scene's render engine to whatever `engine.txt` in the session dir names,
/// the way the Render Properties dropdown does (an RNA set, no wobble spec),
/// and on `look.txt` screenshots the 3D view and writes the luminance of
/// its centre to `seen.txt` (what Blender's own viewport is showing).
const SWITCHER: &str = r#"
import bpy, os
d = os.environ["WOBBLE_SHM_DIR"]
def poll():
    p = os.path.join(d, "engine.txt")
    if os.path.exists(p):
        want = open(p).read().split()
        os.remove(p)
        for e in want:
            try:
                bpy.context.scene.render.engine = e
                print("wobble | test: engine set to", e, flush=True)
                break
            except TypeError:
                continue
    p = os.path.join(d, "look.txt")
    if os.path.exists(p):
        os.remove(p)
        shot = os.path.join(d, "viewport.png")
        win = bpy.context.window_manager.windows[0]
        area = next(a for a in win.screen.areas if a.type == "VIEW_3D")
        with bpy.context.temp_override(window=win, area=area):
            bpy.ops.screen.screenshot_area(filepath=shot)
        img = bpy.data.images.load(shot)
        w, h = img.size
        # Off centre: the 3D cursor sits on the middle pixel.
        i = (int(h * 0.6) * w + int(w * 0.4)) * 4
        px = img.pixels[i:i + 3]
        bpy.data.images.remove(img)
        open(os.path.join(d, "seen.txt"), "w").write("%f" % (sum(px) / 3.0))
    # `leave.txt`: the hand takes the 3D view out of camera view;
    # `ask.txt`: report the view's perspective and Lock Camera to View.
    win = bpy.context.window_manager.windows[0]
    area = next((a for a in win.screen.areas if a.type == "VIEW_3D"), None)
    p = os.path.join(d, "leave.txt")
    if os.path.exists(p) and area is not None:
        os.remove(p)
        area.spaces.active.region_3d.view_perspective = "PERSP"
    p = os.path.join(d, "ask.txt")
    if os.path.exists(p) and area is not None:
        os.remove(p)
        sp = area.spaces.active
        open(os.path.join(d, "told.txt"), "w").write("%s %s" % (sp.region_3d.view_perspective, sp.lock_camera))
    return 0.1
bpy.app.timers.register(poll, persistent=True)
"#;

/// Reported: in live Cycles, switching the engine to EEVEE in Blender
/// froze wobble on the last Cycles sample (and gave Blender its viewport
/// back). The bridge must follow the switch both ways: EEVEE frames come
/// over the ring after it, and Cycles updates resume in wobble's buffer
/// when the user switches back.
#[test]
#[ignore]
fn switching_engines_in_blender_keeps_wobble_live() {
    let exe = blender::executable();
    let ctx = engine::GpuContext::new().unwrap();
    let dir = std::env::temp_dir().join(format!("wobble-cycles-switch-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let make = dir.join("make_scene.py");
    std::fs::write(&make, MAKE_SCENE).unwrap();
    let blend = dir.join("probe.blend");
    std::process::Command::new(&exe)
        .args(["-b", "--factory-startup", "-P"])
        .arg(&make)
        .arg("--")
        .arg(&blend)
        .output()
        .expect("running blender headless");
    assert!(blend.exists(), "scene not written");
    let script = blender::install_script(&dir).unwrap();
    let switcher = dir.join("switcher.py");
    std::fs::write(&switcher, SWITCHER).unwrap();
    let port = live::free_port().unwrap();
    let child = std::process::Command::new(&exe)
        .arg(&blend)
        .arg("--python")
        .arg(&script)
        .arg("--python")
        .arg(&switcher)
        .env("WOBBLE_PORT", port.to_string())
        .env("WOBBLE_SHM_DIR", &dir)
        .spawn()
        .unwrap();
    // A failed assert must not leave Blender holding the test's stdout.
    struct Kill(std::process::Child);
    impl Drop for Kill {
        fn drop(&mut self) {
            let _ = self.0.kill();
        }
    }
    let _blender = Kill(child);
    let s = live::Session::connect(port);

    let t0 = Instant::now();
    let pid = loop {
        assert!(t0.elapsed() < Duration::from_secs(90), "no cycles_fork: is WOBBLE_BLENDER a fork build?");
        if let Some(p) = s.poll().into_iter().find_map(|ev| match ev {
            live::Event::CyclesFork { pid } => Some(pid),
            _ => None,
        }) {
            break p;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut target = CyclesTarget::new(&ctx.device, 4096, 2304, "switch").unwrap();
    s.cycles(target.handle_for(pid).unwrap(), target.size, &target.uuid_hex(), &target.stamp_name);
    let p = blender::BlenderParams {
        engine: blender::BlenderEngine::Cycles,
        quality: Quality::Viewport,
        shading: blender::Shading::Rendered,
        transparent: false,
        ..Default::default()
    };
    s.spec(&p.spec_line(W, H));
    s.frame(4);

    // Ring frames (EEVEE draws) and Cycles updates over `secs`.
    let watch = |target: &mut CyclesTarget, secs: u64| -> (u32, u32, Option<f32>) {
        let t0 = Instant::now();
        let (mut ring, mut cyc, mut centre) = (0u32, 0u32, None);
        while t0.elapsed() < Duration::from_secs(secs) {
            for ev in s.poll() {
                match ev {
                    live::Event::Frame(f) => {
                        ring += 1;
                        if f.fmt == engine::blender::shm::FMT_RGBA32F {
                            let px: &[f32] = bytemuck::cast_slice(&f.pixels);
                            let i = ((f.h / 2 * f.w + f.w / 2) * 4) as usize;
                            centre = Some(px[i]);
                        }
                    }
                    live::Event::FrameRef { .. } | live::Event::SurfaceFrame { .. } => ring += 1,
                    live::Event::Error(e) => eprintln!("  error: {e}"),
                    _ => {}
                }
            }
            if target.poll().is_some() {
                cyc += 1;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        (ring, cyc, centre)
    };

    // What Blender's own 3D view shows at its centre (display-referred).
    let look = || -> f32 {
        let seen = dir.join("seen.txt");
        let _ = std::fs::remove_file(&seen);
        std::fs::write(dir.join("look.txt"), "").unwrap();
        let t0 = Instant::now();
        loop {
            let _ = s.poll();
            if let Ok(v) = std::fs::read_to_string(&seen) {
                if let Ok(v) = v.trim().parse() {
                    return v;
                }
            }
            assert!(t0.elapsed() < Duration::from_secs(10), "no screenshot of the 3D view");
            std::thread::sleep(Duration::from_millis(20));
        }
    };

    let (_, cyc, _) = watch(&mut target, 15);
    eprintln!("cycles: {cyc} updates");
    assert!(cyc > 0, "live Cycles never started");
    // Blender's viewport shows the same render: the lit plane, not the
    // dark world (0.05) or a blank viewport.
    let lum = look();
    eprintln!("blender's viewport centre during live cycles: {lum}");
    assert!(lum > 0.3, "Blender's own viewport is not showing the Cycles render ({lum})");

    std::fs::write(dir.join("engine.txt"), "BLENDER_EEVEE BLENDER_EEVEE_NEXT").unwrap();
    let (ring, _, centre) = watch(&mut target, 10);
    eprintln!("after the switch to EEVEE: {ring} ring frames, centre {centre:?}");
    assert!(ring > 0, "wobble froze: no EEVEE frame after the engine switch");
    if let Some(c) = centre {
        assert!((c - 0.5).abs() < 0.05, "EEVEE centre {c}");
    }

    std::fs::write(dir.join("engine.txt"), "CYCLES").unwrap();
    let (_, cyc, _) = watch(&mut target, 15);
    eprintln!("after the switch back to Cycles: {cyc} updates");
    assert!(cyc > 0, "live Cycles did not resume after switching back");
    let lum = look();
    eprintln!("blender's viewport centre after switching back: {lum}");
    assert!(lum > 0.3, "Blender's own viewport is not showing the Cycles render ({lum})");

    // A tumble report: a tumble in Blender's view must not strand
    // wobble's orbit. Live in camera mode, the view locks the camera to
    // itself (a Blender tumble moves the camera), and if the view leaves
    // camera view anyway, wobble's next orbit brings it back.
    let ask = || -> String {
        let told = dir.join("told.txt");
        let _ = std::fs::remove_file(&told);
        std::fs::write(dir.join("ask.txt"), "").unwrap();
        let t0 = Instant::now();
        loop {
            let _ = s.poll();
            if let Ok(v) = std::fs::read_to_string(&told) {
                if !v.is_empty() {
                    return v;
                }
            }
            assert!(t0.elapsed() < Duration::from_secs(10), "no answer from the 3D view");
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    assert_eq!(ask(), "CAMERA True", "live cycles in camera mode locks the camera to the view");
    std::fs::write(dir.join("leave.txt"), "").unwrap();
    std::thread::sleep(Duration::from_millis(500));
    assert!(ask().starts_with("PERSP"), "the stand-in hand did not leave camera view");
    let eye = blender::ViewState { loc: [0.0, 0.0, 0.0], rot: [1.0, 0.0, 0.0, 0.0], dist: 5.0, lens: 50.0, ortho: false };
    s.view(&eye, false);
    std::thread::sleep(Duration::from_millis(500));
    assert!(ask().starts_with("CAMERA"), "wobble's orbit did not bring the view back to the camera");
    let (_, cyc, _) = watch(&mut target, 10);
    eprintln!("after wobble's orbit: {cyc} updates");
    assert!(cyc > 0, "live Cycles stopped after the orbit");

    s.quit();
    std::thread::sleep(Duration::from_millis(500));
}
