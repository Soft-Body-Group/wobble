//! The Blender bridge against a REAL Blender: launch it on a known scene,
//! ask for frames through the live session, and check the pixels that
//! come back through the ring — the viewport draw and the F12 render both
//! scene-linear, both the right size, both stamped with the frame asked.
//!
//! Ignored: needs Blender (`WOBBLE_BLENDER` or /Applications) and a
//! display, and opens a Blender window for ~20s.
//!
//!     cargo test -p engine --test blender_live -- --ignored --nocapture

use engine::blender::{self, live, Quality, Shading, ViewState};
use std::cell::RefCell;
use std::path::PathBuf;
use std::time::{Duration, Instant};

thread_local! {
    /// The eye as last reported, whatever `wait` was looking for at the
    /// time — a `view` arrives right after hello and would be dropped by
    /// the wait that drains the hello.
    static LAST_VIEW: RefCell<Option<ViewState>> = const { RefCell::new(None) };
    /// Blender's shared surfaces, imported on a wgpu device of our own:
    /// the zero-copy path, read back here only to check the pixels.
    static SURFACES: RefCell<Option<(engine::GpuContext, Vec<blender::gpuimport::SurfaceTexture>)>> = const { RefCell::new(None) };
    /// A device made BEFORE Blender is launched: creating one mid-wait
    /// once took 33s and read as a slow link.
    static GPU: RefCell<Option<engine::GpuContext>> = const { RefCell::new(None) };
}

/// A surface frame as the ring would have delivered it: the pixels read
/// off the shared surface, so every check below works on either path.
fn surface_frame(s: &live::Session, slot: usize, seq: u64, st: blender::shm::Stamp) -> Option<blender::shm::Frame> {
    let px = SURFACES.with(|c| {
        let c = c.borrow();
        let (_, surfaces) = c.as_ref()?;
        surfaces.get(slot)?.read_rows(st.w, st.h)
    });
    s.sack(slot, seq);
    Some(blender::shm::Frame {
        seq,
        frame: st.frame,
        w: st.w,
        h: st.h,
        fmt: st.fmt,
        flags: st.flags,
        scene: st.scene,
        ms: st.ms,
        samples: st.samples,
        pass: st.pass,
        pixels: px?,
    })
}

const W: u32 = 256;
const H: u32 = 128;

/// A 2-unit plane with an emission of exactly 0.5, five units under a
/// 50mm camera: it covers the middle of the frame (half-width 1 of the
/// 1.8 visible), so the centre reads 0.5 and the corners see past it.
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
sc.frame_start, sc.frame_end = 1, 8
bpy.ops.wm.save_as_mainfile(filepath=out)
"#;

fn wait<T>(s: &live::Session, budget: Duration, mut f: impl FnMut(live::Event) -> Option<T>) -> T {
    let t0 = Instant::now();
    loop {
        // The whole batch is looked at before anything returns: a `view`
        // rides right behind the hello, in the same poll.
        let batch = s.poll();
        for ev in &batch {
            if let live::Event::Log(l) = ev {
                eprintln!("  {l}");
            }
            if let live::Event::Error(e) = ev {
                eprintln!("  ERROR {e}");
            }
            if let live::Event::View(v) = ev {
                LAST_VIEW.with(|l| *l.borrow_mut() = Some(*v));
            }
            if let live::Event::Surfaces { ids, w, h } = ev {
                // Import them like the app does; without a GPU here,
                // Blender falls back to the ring and the test goes on.
                let ctx = GPU.with(|g| g.borrow_mut().take()).ok_or_else(|| anyhow::anyhow!("no GPU here"));
                let ok = match ctx {
                    Ok(ctx) => match blender::gpuimport::import_surfaces(&ctx.device, ids, *w, *h) {
                        Ok(t) => {
                            eprintln!("shared surfaces: {} of {w}x{h} imported (zero-copy path) at {:?} into this wait", t.len(), t0.elapsed());
                            SURFACES.with(|c| *c.borrow_mut() = Some((ctx, t)));
                            true
                        }
                        Err(e) => {
                            eprintln!("shared surfaces refused: {e:#}");
                            false
                        }
                    },
                    Err(e) => {
                        eprintln!("no GPU for the surfaces ({e}); the ring serves");
                        false
                    }
                };
                s.surfaces_ok(ok);
            }
        }
        let mut hit = None;
        for ev in batch {
            if hit.is_none() {
                // A surface frame is read back into a ring-shaped frame so
                // the checks below see one kind of picture.
                let ev = match ev {
                    live::Event::SurfaceFrame { slot, seq, stamp } => match {
                        eprintln!("  surface frame {} pass {} slot {slot} at {:?} into this wait", stamp.frame, stamp.pass, t0.elapsed());
                        surface_frame(s, slot, seq, stamp)
                    } {
                        Some(f) => live::Event::Frame(f),
                        None => continue,
                    },
                    other => other,
                };
                hit = f(ev);
            }
        }
        if let Some(v) = hit {
            return v;
        }
        assert!(t0.elapsed() < budget, "waited {budget:?} for nothing");
        assert!(!s.closed(), "session closed while waiting");
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn at(f: &blender::shm::Frame, x: u32, y: u32) -> [f32; 4] {
    let i = (y * f.w + x) as usize * 4;
    match f.fmt {
        blender::shm::FMT_RGBA16F => {
            let px: &[half::f16] = bytemuck::cast_slice(&f.pixels);
            [px[i].to_f32(), px[i + 1].to_f32(), px[i + 2].to_f32(), px[i + 3].to_f32()]
        }
        _ => {
            let px: &[f32] = bytemuck::cast_slice(&f.pixels);
            [px[i], px[i + 1], px[i + 2], px[i + 3]]
        }
    }
}

fn centre(f: &blender::shm::Frame) -> [f32; 4] {
    at(f, f.w / 2, f.h / 2)
}

fn near(a: [f32; 4], want: [f32; 3]) -> bool {
    (0..3).all(|i| (a[i] - want[i]).abs() < 0.02)
}

#[test]
#[ignore]
fn a_real_blender_draws_and_renders_the_scene_camera_into_the_ring() {
    let exe = blender::executable();
    assert!(exe.exists() || exe == PathBuf::from("blender"), "no blender at {}", exe.display());
    if let Ok(ctx) = engine::GpuContext::new() {
        GPU.with(|g| *g.borrow_mut() = Some(ctx));
    }
    let dir = std::env::temp_dir().join(format!("wobble-blender-live-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // 1. The scene, built headless.
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

    // 2. The live session on it.
    let script = blender::install_script(&dir).unwrap();
    let port = live::free_port().unwrap();
    let s = live::Session::launch(&exe, Some(&blend), &script, port, &dir).unwrap();
    let hello = wait(&s, Duration::from_secs(90), |ev| match ev {
        live::Event::Hello { version, w, h, .. } => Some((version, w, h)),
        _ => None,
    });
    eprintln!("hello from blender {} ({}x{})", hello.0, hello.1, hello.2);

    // 3. A viewport draw at the comp's size: scene-linear 0.5, settled.
    let mut p = blender::BlenderParams {
        engine: blender::BlenderEngine::Eevee,
        quality: Quality::Viewport,
        shading: Shading::Rendered,
        transparent: false,
        ..Default::default()
    };
    s.spec(&p.spec_line(W, H));
    s.frame(3);
    let t0 = Instant::now();
    let f = wait(&s, Duration::from_secs(60), |ev| match ev {
        live::Event::Frame(f) if f.frame == 3 && f.settled() => Some(f),
        live::Event::Frame(f) => {
            eprintln!("  refining: frame {} samples {} {:.1}ms", f.frame, f.samples, f.ms);
            None
        }
        _ => None,
    });
    eprintln!("viewport frame 3 settled in {:.0}ms, {} samples, draw {:.1}ms", t0.elapsed().as_millis(), f.samples, f.ms);
    assert_eq!((f.w, f.h), (W, H), "drawn at the spec's size");
    let c = centre(&f);
    assert!((c[0] - 0.5).abs() < 0.01 && (c[1] - 0.5).abs() < 0.01 && (c[2] - 0.5).abs() < 0.01, "viewport centre {c:?}");
    assert!((c[3] - 1.0).abs() < 0.01, "opaque film: alpha {c:?}");
    assert!(!f.is_final());

    // 4. The same frame as a final render, read off the viewer node.
    p.quality = Quality::Final;
    s.spec(&p.spec_line(W, H));
    s.frame(4);
    let t0 = Instant::now();
    let f = wait(&s, Duration::from_secs(120), |ev| match ev {
        live::Event::Frame(f) if f.frame == 4 && f.is_final() => Some(f),
        _ => None,
    });
    eprintln!("final frame 4 in {:.0}ms", t0.elapsed().as_millis());
    assert_eq!((f.w, f.h), (W, H), "rendered at the spec's size");
    let c = centre(&f);
    assert!((c[0] - 0.5).abs() < 0.01 && (c[1] - 0.5).abs() < 0.01 && (c[2] - 0.5).abs() < 0.01, "final centre {c:?}");

    // 5. Transparent film: the world is alpha 0 past the plane, 1 on it.
    p.transparent = true;
    p.quality = Quality::Viewport;
    s.spec(&p.spec_line(W, H));
    s.frame(5);
    let f = wait(&s, Duration::from_secs(60), |ev| match ev {
        live::Event::Frame(f) if f.frame == 5 && f.settled() => Some(f),
        _ => None,
    });
    assert!((centre(&f)[0] - 0.5).abs() < 0.01);
    let corner = at(&f, 4, 4);
    assert!(corner[3] < 0.01, "past the plane the world is alpha 0: {corner:?}");

    // 6. The eye: hello reported the camera (identity rotation five units
    // up, so the pivot sits CARD_DIST below the eye).
    let eye = LAST_VIEW.with(|l| *l.borrow()).expect("hello reported the eye");
    eprintln!("eye {eye:?}");
    assert!((eye.dist - 10.0).abs() < 1e-3 && (eye.loc[2] - (-5.0)).abs() < 1e-3, "{eye:?}");
    assert!((eye.rot[0] - 1.0).abs() < 1e-3, "identity rotation: {eye:?}");

    // 7. A texture in, on a card behind the plane (cards on): the corner,
    // which saw the world, now sees the plate; the centre still the plane.
    p.inputs = 1;
    p.cards = true;
    s.spec(&p.spec_line(W, H));
    let ring = blender::shm::Source::for_writer("wobtest_in", &dir, 0);
    let mut w = blender::shm::Writer::create(&ring, 2, blender::shm::Writer::bytes_for(W, H, blender::shm::FMT_RGBA16F).unwrap()).unwrap();
    let plate: Vec<u8> = (0..W * H)
        .flat_map(|_| {
            [0.25f32, 0.5, 0.75, 1.0].into_iter().flat_map(|v| half::f16::from_f32(v).to_le_bytes())
        })
        .collect();
    let stamp = blender::shm::Stamp { w: W, h: H, fmt: blender::shm::FMT_RGBA16F, flags: blender::shm::FLAG_SETTLED, ..Default::default() };
    let (slot, seq) = w.write(stamp, &plate).unwrap();
    s.shm(&w.source);
    s.tex(0, slot, seq);
    s.frame(6);
    let t0 = Instant::now();
    let f = wait(&s, Duration::from_secs(60), |ev| match ev {
        live::Event::Frame(f) if f.frame == 6 && f.settled() && near(at(&f, 4, 4), [0.25, 0.5, 0.75]) => Some(f),
        live::Event::Frame(f) if f.frame == 6 => {
            eprintln!("  frame 6 corner {:?} (texture not there yet)", at(&f, 4, 4));
            None
        }
        _ => None,
    });
    eprintln!("texture round trip in {:.0}ms", t0.elapsed().as_millis());
    assert!((centre(&f)[0] - 0.5).abs() < 0.01, "the plane still covers the centre: {:?}", centre(&f));

    // 8. wobble moves the eye (camera mode): back off to 20 units and the
    // plane shrinks — a pixel 20% of the width out from the centre (the
    // plane reaches 28% at five units, 9% at fifteen), on the plane
    // before, now sees the card.
    let x30 = W / 2 + (W as f32 * 0.2) as u32;
    assert!(near(at(&f, x30, H / 2), [0.5, 0.5, 0.5]), "before the move: {:?}", at(&f, x30, H / 2));
    // The card rides with the camera (it would sit between the two), so
    // it goes away for this: past the plane is the transparent world.
    p.cards = false;
    s.spec(&p.spec_line(W, H));
    let far = ViewState { dist: 20.0, ..eye };
    s.view(&far, true);
    s.view(&far, false);
    s.frame(7);
    let f = wait(&s, Duration::from_secs(60), |ev| match ev {
        live::Event::Frame(f) if f.frame == 7 && f.settled() && at(&f, x30, H / 2)[3] < 0.01 => Some(f),
        live::Event::Frame(f) if f.frame == 7 => {
            eprintln!("  frame 7 at 20%: {:?}", at(&f, x30, H / 2));
            None
        }
        _ => None,
    });
    assert!((centre(&f)[0] - 0.5).abs() < 0.01, "centre still the plane: {:?}", centre(&f));

    // 9. Free mode: the 3D view's own eye. Put it where the camera was
    // and the picture must match the camera's (same maths, same size).
    p.view = blender::View::Free;
    s.spec(&p.spec_line(W, H));
    let t0 = Instant::now();
    let eye_view = loop {
        let before = LAST_VIEW.with(|l| *l.borrow());
        let _ = wait(&s, Duration::from_secs(10), |ev| match ev {
            live::Event::View(v) => Some(v),
            _ => None,
        });
        let now = LAST_VIEW.with(|l| *l.borrow());
        if now != before || t0.elapsed() > Duration::from_secs(10) {
            break now.unwrap();
        }
    };
    eprintln!("free eye as Blender left it: {eye_view:?}");
    // A free view sees twice as wide for the same lens: 100mm free is the
    // 50mm camera's field.
    s.view(&ViewState { lens: 100.0, ..eye }, false);
    s.frame(8);
    let f = wait(&s, Duration::from_secs(60), |ev| match ev {
        live::Event::Frame(f) if f.frame == 8 && f.settled() => Some(f),
        _ => None,
    });
    assert!((centre(&f)[0] - 0.5).abs() < 0.01, "free view centre: {:?}", centre(&f));
    let edge = at(&f, x30, H / 2);
    assert!(near(edge, [0.5, 0.5, 0.5]), "free view at 30%: plane again (eye at 10 like the camera): {edge:?}");

    // The scene is dirty (cards, images), so a polite quit stops at
    // Blender's save dialog; this Blender is the test's, so it is killed
    // once that shows.
    s.quit();
    let t0 = Instant::now();
    while !s.closed() && t0.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(50));
    }
    eprintln!("closed: {}", s.closed());
    if !s.closed() {
        s.kill();
    }
}
