//! The app half of the Blender node: one live session per node, what it
//! has handed over, the panel, and the playhead running both ways.
//!
//! The engine half (`engine::blender`) owns the process, the socket and
//! the pixel ring; this module decides WHEN to ask for a frame, what to do
//! with one that lands, and what the user sees. Three rules:
//!
//! - **The layer holds one picture, stamped.** A frame that lands goes
//!   straight into the node's GPU layer (`Compositor::blender_upload`)
//!   with the Blender frame and scene version it was drawn at. The panel
//!   reads that stamp back and says, in words, whether the picture on the
//!   canvas is the frame the playhead asks for.
//! - **Settled frames are cached, per scene version.** A scrub back over
//!   frames Blender already finished costs one upload and no round trip;
//!   Blender still moves its own playhead (`cached: true` in the request)
//!   but skips the draw. Any edit in Blender bumps the version and empties
//!   the cache, so a stale frame can never be shown as current.
//! - **Nothing here blocks.** Events are drained once a redraw; an upload
//!   is a `write_texture`. The one stall is exporting an input texture —
//!   a full render and a readback — and the node is red for it.

use crate::widgets;
use egui::Ui;
use engine::blender::{self, gpuimport, live, shm, BlenderParams, Quality, ViewState};
use engine::compositor::BlenderStamp;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Fingers still this long end a two-finger gesture on the eye.
const PAD_IDLE: std::time::Duration = std::time::Duration::from_millis(150);

/// Settled frames kept per node, by bytes. Two seconds of 2K at 8 bytes a
/// pixel; scaled down with the frame size, never below one frame.
pub const CACHE_BYTES: usize = 48 * 2048 * 1080 * 8;
/// Panel log lines kept.
const LOG_LINES: usize = 3;
/// The panel's footprint on the properties grid: two tiles wide (buttons
/// and names side by side) and two tall (the rows above plus the status).
#[allow(dead_code)]
pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

/// What `hello` told us about the Blender on the other end.
#[derive(Clone, Debug, PartialEq)]
pub struct Hello {
    pub version: String,
    pub w: u32,
    pub h: u32,
    pub fps: f32,
    pub blend: String,
    pub frame: i64,
    pub start: i64,
    pub end: i64,
}

/// The last frame that landed, for the panel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landed {
    pub frame: i64,
    pub scene: u64,
    pub settled: bool,
    pub is_final: bool,
    pub ms: f32,
    pub samples: u32,
    pub w: u32,
    pub h: u32,
}

/// A settled frame kept for a scrub back, one per (frame, pass).
struct Cached {
    w: u32,
    h: u32,
    fmt: u32,
    is_final: bool,
    pixels: Arc<Vec<u8>>,
}

/// A settled frame copied off the GPU-path ring for the cache, by a
/// helper thread, so the display path never waits on a memcpy.
struct CacheCopy {
    frame: i64,
    pass: u8,
    scene: u64,
    w: u32,
    h: u32,
    fmt: u32,
    is_final: bool,
    pixels: Vec<u8>,
}

/// One node's live session and everything the app remembers about it.
pub struct Live {
    pub session: live::Session,
    pub hello: Option<Hello>,
    /// Blender's scene version as last announced; frames stamped older
    /// than this are stale and never shown over a current one.
    pub scene: u64,
    /// The Blender frame last requested (so a redraw with nothing new
    /// sends nothing), and the spec last sent.
    pub sent: Option<i64>,
    pub spec_sent: Option<String>,
    pub landed: Option<Landed>,
    /// Frames Blender announced that could not be read (lapped); shown
    /// in the panel because a rising count means the ring is too small.
    pub dropped: u64,
    pub frames_in: u64,
    pub log: Vec<String>,
    pub error: Option<String>,
    pub closed: Option<String>,
    cache: HashMap<(i64, u8), Cached>,
    cache_bytes: usize,
    /// The ring wobble's textures go out on, created on first export.
    tex_ring: Option<shm::Writer>,
    /// Per input port: what was last exported (subtree key, tick).
    pub tex_sent: HashMap<usize, (u64, u64)>,
    /// A frame Blender's own playhead moved to, not yet applied to ours.
    pub follow: Option<i64>,
    /// The 3D cards' export (`scene3d`): the scene version last asked
    /// for, and the file Blender last wrote.
    pub scene3d_asked: Option<u64>,
    pub scene3d_file: Option<String>,
    /// A new export landed and the comp has not redrawn for it yet.
    pub scene3d_fresh: bool,
    /// Where the session's files live (script, rings).
    pub dir: PathBuf,
    /// The eye as Blender last reported it, or as wobble's hand left it.
    pub view: Option<ViewState>,
    /// A wobble drag is on the view: its steps go out as `moving`, and
    /// Blender's own reports are ignored until it ends.
    pub nav_active: bool,
    /// The last two-finger step on the eye: a trackpad has no release, so
    /// the gesture ends (full samples) once the fingers have been still
    /// for `PAD_IDLE`.
    pad_at: Option<web_time::Instant>,
    /// The ring on the GPU (`gpuimport`): frames come by reference and
    /// land as one blit. `None` = the CPU path.
    gpu: Option<Arc<gpuimport::SharedRing>>,
    /// Blender's frames as shared textures (IOSurfaces the native module
    /// blits into): no readback anywhere. `None` = the ring.
    surfaces: Option<Arc<Vec<gpuimport::SurfaceTexture>>>,
    /// The app's own view of the ring, for the cache copies on the GPU
    /// path.
    ring: Option<Arc<shm::Reader>>,
    /// Settled frames the helper thread copied for the cache.
    cache_rx: Option<std::sync::mpsc::Receiver<CacheCopy>>,
    cache_tx: Option<std::sync::mpsc::Sender<CacheCopy>>,
    /// Frames that went straight to the GPU.
    pub blits: u64,
    /// Texture rings made so far (each size is a fresh name).
    tex_rings: usize,
    /// Live Cycles (a wobble-fork Blender): the buffer its CUDA writes
    /// into, and Blender's pid to hand the handle to.
    pub cycles: Option<blender::cycles::CyclesTarget>,
    cycles_pid: Option<u32>,
    /// Blender said it is stock Blender on Cycles: the popup explaining
    /// that live Cycles needs the custom build is up until dismissed.
    cycles_notice: bool,
    /// Updates landed from it, for the panel.
    pub cycles_updates: u64,
}

/// A frame for the canvas: pixels to upload into the node's layer, and
/// the stamp they carry.
pub struct Landing {
    /// 0 = the combined picture, else a `blender::Pass` id.
    pub pass: u8,
    pub w: u32,
    pub h: u32,
    pub fmt: u32,
    pub pixels: Arc<Vec<u8>>,
    pub stamp: BlenderStamp,
}

/// Where a frame to blit sits on the GPU.
pub enum BlitSource {
    /// The mapped ring: a buffer, the slot's pixels at `offset`.
    Ring { ring: Arc<gpuimport::SharedRing>, offset: u64, stride: u32 },
    /// Blender's shared textures: the slot IS a texture.
    Surface(Arc<Vec<gpuimport::SurfaceTexture>>),
}

/// A frame to blit into the node's layer, GPU to GPU.
pub struct Blit {
    pub source: BlitSource,
    pub slot: usize,
    pub pass: u8,
    /// The frame's size and format, as stamped.
    pub stamp_meta: shm::Stamp,
    pub stamp: BlenderStamp,
    /// Runs once the GPU is done with the slot.
    pub done: Box<dyn FnOnce() + Send + 'static>,
}

impl Live {
    pub fn new(session: live::Session, dir: PathBuf) -> Live {
        Live {
            session,
            hello: None,
            scene: 0,
            sent: None,
            spec_sent: None,
            landed: None,
            dropped: 0,
            frames_in: 0,
            log: Vec::new(),
            error: None,
            closed: None,
            cache: HashMap::new(),
            cache_bytes: 0,
            tex_ring: None,
            tex_sent: HashMap::new(),
            follow: None,
            scene3d_asked: None,
            scene3d_file: None,
            scene3d_fresh: false,
            dir,
            view: None,
            nav_active: false,
            pad_at: None,
            gpu: None,
            surfaces: None,
            ring: None,
            cache_rx: None,
            cache_tx: None,
            blits: 0,
            tex_rings: 0,
            cycles: None,
            cycles_pid: None,
            cycles_notice: false,
            cycles_updates: 0,
        }
    }

    /// Frames land on the GPU without a CPU copy.
    #[allow(dead_code)]
    pub fn gpu_path(&self) -> bool {
        self.gpu.is_some() || self.surfaces.is_some()
    }

    /// Frames never leave the GPU at all (shared textures).
    #[allow(dead_code)]
    pub fn texture_path(&self) -> bool {
        self.surfaces.is_some()
    }

    /// A step of Blender's hands over the picture: orbit (MMB), pan
    /// (shift+MMB) or dolly (ctrl+MMB / wheel), in screen pixels. Applied
    /// to the eye here and sent at once, flagged `moving` so Blender draws
    /// at one sample until `nav_end`.
    pub fn nav(&mut self, params: &BlenderParams, g: Nav, format: (u32, u32), px_high: f32) {
        let mut v = self.view.unwrap_or_default();
        let camera = params.view == blender::View::Camera;
        match g {
            Nav::Orbit(dx, dy) => v.orbit(dx, dy),
            Nav::Pan(dx, dy) => {
                let upp = v.units_per_px(format.0, format.1, camera, px_high);
                v.pan(dx, dy, upp);
            }
            Nav::Dolly(factor) => v.dolly(factor),
        }
        self.view = Some(v);
        self.nav_active = true;
        if self.up() {
            self.session.view(&v, true);
        }
    }

    /// A step from two fingers (or a wheel): as `nav`, and the gesture
    /// stays open until the fingers rest (`pad_idle`).
    pub fn nav_pad(&mut self, params: &BlenderParams, g: Nav, format: (u32, u32), px_high: f32) {
        self.nav(params, g, format, px_high);
        self.pad_at = Some(web_time::Instant::now());
    }

    /// Ends a two-finger gesture whose fingers have rested; true while one
    /// is still open (the caller keeps frames coming to see it end).
    pub fn pad_idle(&mut self) -> bool {
        match self.pad_at {
            Some(t) if t.elapsed() >= PAD_IDLE => {
                self.pad_at = None;
                self.nav_end();
                false
            }
            Some(_) => true,
            None => false,
        }
    }

    /// The hand came off: the same eye once more, at full samples.
    pub fn nav_end(&mut self) {
        if !self.nav_active {
            return;
        }
        self.nav_active = false;
        if let Some(v) = self.view {
            if self.up() {
                self.session.view(&v, false);
            }
        }
    }

    fn note(&mut self, line: String) {
        self.log.push(line);
        if self.log.len() > LOG_LINES {
            self.log.remove(0);
        }
    }

    /// Drain what Blender sent since the last redraw. Returns the
    /// landings the app should upload, in order (usually one), and the
    /// frames to blit by reference. `device` is for importing a ring the
    /// moment Blender announces it.
    pub fn poll(&mut self, params: &BlenderParams, spec: &str, device: &wgpu::Device) -> (Vec<Landing>, Vec<Blit>) {
        let mut out = Vec::new();
        let mut blits = Vec::new();
        // The fork's Cycles buffer follows the spec's size.
        self.bind_cycles(spec, device);
        // Settled frames the helper thread copied since last time.
        if let Some(rx) = &self.cache_rx {
            let got: Vec<CacheCopy> = rx.try_iter().collect();
            for c in got {
                if c.scene == self.scene {
                    self.remember(c.frame, c.pass, c.w, c.h, c.fmt, c.is_final, Arc::new(c.pixels));
                }
            }
        }
        for ev in self.session.poll() {
            match ev {
                live::Event::Ring(src) => {
                    self.gpu = None;
                    self.ring = shm::Reader::open(&src).ok().map(Arc::new);
                    // The GPU path blits float32 as is, which the fused
                    // pass can only sample on a GPU that filters float32;
                    // elsewhere the CPU path converts to half on the way.
                    let import = if device.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
                        gpuimport::import(device, &src)
                    } else {
                        Err(anyhow::anyhow!("this GPU does not filter float32 textures"))
                    };
                    match import {
                        Ok(r) => {
                            self.gpu = Some(Arc::new(r));
                            self.session.set_gpu(true);
                            self.note("ring on the GPU: frames land as one blit".into());
                        }
                        Err(e) => {
                            self.session.set_gpu(false);
                            self.note(format!("ring on the CPU path ({e})"));
                        }
                    }
                    if self.cache_tx.is_none() {
                        let (tx, rx) = std::sync::mpsc::channel();
                        self.cache_tx = Some(tx);
                        self.cache_rx = Some(rx);
                    }
                }
                live::Event::Surfaces { ids, w, h } => {
                    self.surfaces = None;
                    let import = if device.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
                        gpuimport::import_surfaces(device, &ids, w, h)
                    } else {
                        Err(anyhow::anyhow!("this GPU does not filter float32 textures"))
                    };
                    match import {
                        Ok(t) => {
                            self.surfaces = Some(Arc::new(t));
                            self.session.surfaces_ok(true);
                            self.note(format!("frames as shared textures ({} surfaces, {w}x{h}): no readback", ids.len()));
                        }
                        Err(e) => {
                            self.session.surfaces_ok(false);
                            self.note(format!("shared textures refused ({e}); the ring serves"));
                        }
                    }
                    if self.cache_tx.is_none() {
                        let (tx, rx) = std::sync::mpsc::channel();
                        self.cache_tx = Some(tx);
                        self.cache_rx = Some(rx);
                    }
                }
                live::Event::SurfaceFrame { slot, seq, stamp: st } => {
                    self.frames_in += 1;
                    if st.scene > self.scene {
                        self.scene = st.scene;
                        self.cache.clear();
                        self.cache_bytes = 0;
                    }
                    let settled = st.flags & shm::FLAG_SETTLED != 0;
                    let is_final = st.flags & shm::FLAG_FINAL != 0;
                    let Some(surfaces) = self.surfaces.clone().filter(|s| slot < s.len()) else {
                        self.session.sack(slot, seq);
                        continue;
                    };
                    if st.scene < self.scene {
                        self.session.sack(slot, seq);
                        continue;
                    }
                    if st.pass == 0 {
                        self.landed = Some(Landed {
                            frame: st.frame,
                            scene: st.scene,
                            settled,
                            is_final,
                            ms: st.ms,
                            samples: st.samples,
                            w: st.w,
                            h: st.h,
                        });
                    }
                    let parts = if settled { 2 } else { 1 };
                    let pending = Arc::new(std::sync::atomic::AtomicUsize::new(parts));
                    let ack = Arc::new(self.session.acker(true));
                    if settled {
                        if let Some(tx) = self.cache_tx.clone() {
                            let (surfaces, pending, ack) = (surfaces.clone(), pending.clone(), ack.clone());
                            std::thread::spawn(move || {
                                if let Some(px) = surfaces[slot].read_rows(st.w, st.h) {
                                    let _ = tx.send(CacheCopy {
                                        frame: st.frame,
                                        pass: st.pass,
                                        scene: st.scene,
                                        w: st.w,
                                        h: st.h,
                                        fmt: st.fmt,
                                        is_final,
                                        pixels: px,
                                    });
                                }
                                if pending.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                                    ack(slot, seq);
                                }
                            });
                        }
                    }
                    self.blits += 1;
                    blits.push(Blit {
                        source: BlitSource::Surface(surfaces),
                        slot,
                        pass: st.pass,
                        stamp_meta: st,
                        stamp: BlenderStamp { frame: st.frame, scene: st.scene, settled, is_final },
                        done: Box::new(move || {
                            if pending.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                                ack(slot, seq);
                            }
                        }),
                    });
                }
                live::Event::FrameRef { slot, desc } => {
                    self.frames_in += 1;
                    let st = desc.stamp;
                    if st.scene > self.scene {
                        self.scene = st.scene;
                        self.cache.clear();
                        self.cache_bytes = 0;
                    }
                    let (Some(gpu), Some(ring)) = (self.gpu.clone(), self.ring.clone()) else {
                        self.session.ack(slot, desc.seq);
                        continue;
                    };
                    if st.scene < self.scene {
                        self.session.ack(slot, desc.seq);
                        continue;
                    }
                    if st.pass == 0 {
                        self.landed = Some(Landed {
                            frame: st.frame,
                            scene: st.scene,
                            settled: desc.settled(),
                            is_final: desc.is_final(),
                            ms: st.ms,
                            samples: st.samples,
                            w: st.w,
                            h: st.h,
                        });
                    }
                    // The slot is acked once BOTH the GPU blit and (for a
                    // settled frame) the cache copy are done with it.
                    let parts = if desc.settled() { 2 } else { 1 };
                    let pending = Arc::new(std::sync::atomic::AtomicUsize::new(parts));
                    let ack = Arc::new(self.session.acker(false));
                    if desc.settled() {
                        if let Some(tx) = self.cache_tx.clone() {
                            let (ring, pending, ack) = (ring.clone(), pending.clone(), ack.clone());
                            std::thread::spawn(move || {
                                if let Some(f) = ring.read(slot) {
                                    if f.seq == desc.seq {
                                        let _ = tx.send(CacheCopy {
                                            frame: f.frame,
                                            pass: f.pass,
                                            scene: f.scene,
                                            w: f.w,
                                            h: f.h,
                                            fmt: f.fmt,
                                            is_final: f.is_final(),
                                            pixels: f.pixels,
                                        });
                                    }
                                }
                                if pending.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                                    ack(slot, desc.seq);
                                }
                            });
                        }
                    }
                    self.blits += 1;
                    blits.push(Blit {
                        source: BlitSource::Ring { ring: gpu, offset: desc.offset as u64, stride: desc.stride },
                        slot,
                        pass: st.pass,
                        stamp_meta: st,
                        stamp: BlenderStamp {
                            frame: st.frame,
                            scene: st.scene,
                            settled: desc.settled(),
                            is_final: desc.is_final(),
                        },
                        done: Box::new(move || {
                            if pending.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                                ack(slot, desc.seq);
                            }
                        }),
                    });
                }
                live::Event::Hello { version, w, h, fps, blend, frame, start, end } => {
                    let broken = broken_release(&version);
                    self.note(format!("blender {version} — {w}x{h} @ {fps}fps"));
                    self.hello = Some(Hello { version, w, h, fps, blend, frame, start, end });
                    // A (re)connect: everything is sent again.
                    self.spec_sent = None;
                    self.sent = None;
                    self.tex_sent.clear();
                    self.error = broken;
                    self.session.spec(spec);
                    self.spec_sent = Some(spec.to_string());
                }
                live::Event::Frame(f) => {
                    self.frames_in += 1;
                    if f.scene > self.scene {
                        // Blender's version ran ahead of its announcement:
                        // trust the frame's stamp.
                        self.scene = f.scene;
                        self.cache.clear();
                        self.cache_bytes = 0;
                    }
                    if f.scene < self.scene {
                        continue; // drawn before an edit that has since landed
                    }
                    let stamp = BlenderStamp {
                        frame: f.frame,
                        scene: f.scene,
                        settled: f.settled(),
                        is_final: f.is_final(),
                    };
                    if f.pass == 0 {
                        self.landed = Some(Landed {
                            frame: f.frame,
                            scene: f.scene,
                            settled: f.settled(),
                            is_final: f.is_final(),
                            ms: f.ms,
                            samples: f.samples,
                            w: f.w,
                            h: f.h,
                        });
                    }
                    let (settled, is_final, w, h, fmt, frame, pass) =
                        (f.settled(), f.is_final(), f.w, f.h, f.fmt, f.frame, f.pass);
                    let pixels = Arc::new(f.pixels);
                    if settled {
                        self.remember(frame, pass, w, h, fmt, is_final, pixels.clone());
                    }
                    out.push(Landing { pass, w, h, fmt, pixels, stamp });
                }
                live::Event::Dropped => self.dropped += 1,
                live::Event::FrameChanged(f) => {
                    if params.sync.follows_blender() {
                        self.follow = Some(f);
                    }
                }
                live::Event::View(v) => {
                    // Blender's report; a drag in progress here outranks it
                    // (it would be our own move coming back, one step late).
                    if !self.nav_active {
                        self.view = Some(v);
                    }
                }
                live::Event::Scene(v) => {
                    if v > self.scene {
                        self.scene = v;
                        self.cache.clear();
                        self.cache_bytes = 0;
                    }
                }
                live::Event::Saved { path } => self.note(format!("saved {path}")),
                live::Event::Scene3d { path } => {
                    self.scene3d_file = Some(path);
                    self.scene3d_fresh = true;
                }
                live::Event::Error(e) => {
                    self.note(format!("error: {e}"));
                    self.error = Some(e);
                }
                live::Event::CyclesFork { pid } => {
                    self.cycles_pid = Some(pid);
                    self.note(format!("blender is a wobble fork (pid {pid}): live Cycles available"));
                    self.bind_cycles(spec, device);
                }
                live::Event::CyclesStock => {
                    self.note("cycles: no live cycles in stock blender; the picture is the material preview".into());
                    self.cycles_notice = self.cycles_pid.is_none();
                }
                live::Event::CyclesBound { ok, msg } => {
                    if ok {
                        self.note("cycles renders straight into wobble".into());
                    } else {
                        self.note(format!("cycles target refused: {msg}"));
                        self.cycles = None;
                    }
                }
                live::Event::Pong => {}
                live::Event::Log(l) => self.note(l),
                live::Event::Closed(why) => {
                    self.note(format!("closed: {why}"));
                    self.closed = Some(why);
                }
            }
        }
        (out, blits)
    }

    /// (Re)make the shared Cycles buffer at the spec's size and hand it to
    /// Blender. A no-op without a fork on the other end, or when the
    /// buffer already fits.
    fn bind_cycles(&mut self, spec: &str, device: &wgpu::Device) {
        let Some(pid) = self.cycles_pid else { return };
        // The spec is one flat JSON line (`BlenderParams::spec_line`).
        let dim = |k: &str| -> u32 {
            spec.split(&format!("\"{k}\":"))
                .nth(1)
                .and_then(|r| r.split(|c: char| !c.is_ascii_digit()).next())
                .and_then(|s| s.parse().ok())
                .unwrap_or(0)
        };
        // Room for the comp format AND a live viewport render, which is as
        // big as the region the camera frame fills (up to 4K here).
        let (w, h) = (dim("w"), dim("h"));
        let (w, h) = if w == 0 || h == 0 { (0, 0) } else { (w.max(4096), h.max(2304)) };
        if w == 0 || h == 0 || self.cycles.as_ref().is_some_and(|t| t.width == w && t.height == h) {
            return;
        }
        self.tex_rings += 1;
        match blender::cycles::CyclesTarget::new(device, w, h, &format!("{}_{}", self.session.port, self.tex_rings)) {
            Ok(t) => match t.handle_for(pid) {
                Ok(handle) => {
                    self.session.cycles(handle, t.size, &t.uuid_hex(), &t.stamp_name);
                    self.note(format!("cycles buffer {w}x{h} shared with blender"));
                    self.cycles = Some(t);
                }
                Err(e) => self.note(format!("cycles buffer: {e:#}")),
            },
            Err(e) => self.note(format!("cycles buffer: {e:#}")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn remember(&mut self, frame: i64, pass: u8, w: u32, h: u32, fmt: u32, is_final: bool, pixels: Arc<Vec<u8>>) {
        let n = pixels.len();
        if let Some(old) = self.cache.remove(&(frame, pass)) {
            self.cache_bytes -= old.pixels.len();
        }
        // Evict the frames farthest from this one until it fits; a single
        // frame always fits, whatever the budget.
        while self.cache_bytes + n > CACHE_BYTES && !self.cache.is_empty() {
            let far = *self.cache.keys().max_by_key(|k| ((k.0 - frame).abs(), k.1)).unwrap();
            let gone = self.cache.remove(&far).unwrap();
            self.cache_bytes -= gone.pixels.len();
        }
        self.cache_bytes += n;
        self.cache.insert((frame, pass), Cached { w, h, fmt, is_final, pixels });
    }

    /// The settled pictures kept for `frame` at this scene version (the
    /// combined and every pass that landed), empty if none.
    pub fn cached(&self, frame: i64) -> Vec<Landing> {
        let mut out: Vec<Landing> = self
            .cache
            .iter()
            .filter(|((f, _), _)| *f == frame)
            .map(|((_, pass), c)| Landing {
                pass: *pass,
                w: c.w,
                h: c.h,
                fmt: c.fmt,
                pixels: c.pixels.clone(),
                stamp: BlenderStamp { frame, scene: self.scene, settled: true, is_final: c.is_final },
            })
            .collect();
        out.sort_by_key(|l| l.pass);
        out
    }

    /// The combined picture is cached for `frame`.
    pub fn has_cached(&self, frame: i64) -> bool {
        self.cache.contains_key(&(frame, 0))
    }

    /// The session is up and answering.
    pub fn up(&self) -> bool {
        self.session.connected() && self.closed.is_none()
    }

    /// Send the spec if it changed, then the frame if it changed. `cached`
    /// says the frame is already on the canvas from the cache, so Blender
    /// moves its playhead and skips the draw.
    pub fn drive(&mut self, params: &BlenderParams, spec: &str, frame: i64, cached: bool) {
        if !self.up() {
            return;
        }
        if self.spec_sent.as_deref() != Some(spec) {
            self.session.spec(spec);
            self.spec_sent = Some(spec.to_string());
            // A new spec redraws whatever frame is current: resend it.
            self.sent = None;
        }
        if params.sync.drives_blender() && self.sent != Some(frame) {
            self.session.frame_ex(frame, cached);
            self.sent = Some(frame);
        }
    }

    /// Hand a picture to Blender as input port `port`. `pixels` is
    /// rgba16f, rows top-down, `w * h * 8` bytes.
    pub fn send_texture(&mut self, port: usize, w: u32, h: u32, pixels: &[u8]) {
        if !self.up() {
            return;
        }
        let Some(need) = shm::Writer::bytes_for(w, h, shm::FMT_RGBA16F) else { return };
        let fresh = match &self.tex_ring {
            Some(r) => r.slot_bytes() < need,
            None => true,
        };
        if fresh {
            // Shared memory on Unix (no file, no writeback), a temp file
            // elsewhere. A new size is a new name: the old ring may still
            // be mapped in Blender for a moment.
            let name = format!("wobin_{}_{}", self.session.port, self.tex_sent.len() + self.tex_rings);
            self.tex_rings += 1;
            let src = shm::Source::for_writer(&name, &self.dir, need);
            match shm::Writer::create(&src, 3, need) {
                Ok(w) => {
                    self.session.shm(&w.source);
                    self.tex_ring = Some(w);
                }
                Err(e) => {
                    self.note(format!("texture ring: {e:#}"));
                    return;
                }
            }
        }
        let ring = self.tex_ring.as_mut().unwrap();
        let stamp = shm::Stamp { w, h, fmt: shm::FMT_RGBA16F, flags: shm::FLAG_SETTLED, ..Default::default() };
        match ring.write(stamp, pixels) {
            Ok((slot, seq)) => self.session.tex(port, slot, seq),
            Err(e) => self.note(format!("texture {port}: {e:#}")),
        }
    }
}

// The texture ring unlinks itself when the writer drops with the session.

/// The popup for a Cycles scene on stock Blender. True once dismissed.
fn cycles_notice(ctx: &egui::Context, node: u64) -> bool {
    let mut done = false;
    let r = egui::Modal::new(egui::Id::new(("blender-cycles-notice", node))).show(ctx, |ui| {
        ui.set_max_width(360.0);
        ui.heading("No live Cycles in stock Blender");
        ui.add_space(6.0);
        ui.label(
            "Live Cycles needs wobble's custom Blender build, which is not available yet. \
             With this Blender the live picture is the Material Preview (EEVEE), not Cycles.",
        );
        ui.add_space(4.0);
        ui.label(
            "For real Cycles pixels, set the node's picture to final render: every frame is then a \
             full Cycles render (slow; a Bake below the node turns it into files).",
        );
        ui.add_space(8.0);
        done = ui.button("ok").clicked();
    });
    done || r.should_close()
}

/// Said in words when the Blender on the other end is 5.1, whose
/// offscreen EEVEE draw leaves every object out (measured on Windows,
/// OpenGL and Vulkan alike: the world alone, while F12 and 5.2.1 draw the
/// scene), so its viewport frames are a picture of nothing.
fn broken_release(version: &str) -> Option<String> {
    let mut n = version.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty());
    let major: u32 = n.next()?.parse().ok()?;
    let minor: u32 = n.next()?.parse().ok()?;
    ((major, minor) == (5, 1)).then(|| {
        format!("blender {major}.{minor} draws an empty viewport offscreen — use 5.2 (final renders still work)")
    })
}

/// One step of Blender's hands on the picture, in screen pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Nav {
    Orbit(f32, f32),
    Pan(f32, f32),
    /// Multiplicative distance: above 1 backs off.
    Dolly(f32),
}

/// What the panel asks the app to do, after the node borrow ends.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Launch Blender on the node's `.blend` (or a fresh scene).
    Open,
    /// Join a Blender already listening (the addon's connect) on a port.
    Connect(u16),
    /// Drop the session; Blender keeps running.
    Detach,
    /// Ask Blender to close.
    Quit,
    /// Re-send the current frame (a picture that looks stale).
    Refresh,
    /// Close the "no live Cycles in stock Blender" popup.
    DismissCyclesNotice,
}

/// A snapshot of a session for the panel: owned, because the panel runs
/// inside the node borrow and the session lives on the app.
#[derive(Clone, Debug)]
pub struct LiveInfo {
    pub connected: bool,
    pub closed: Option<String>,
    pub age: f32,
    pub hello: Option<Hello>,
    pub landed: Option<Landed>,
    pub error: Option<String>,
    pub frames_in: u64,
    pub dropped: u64,
    pub cached: usize,
    pub log: Vec<String>,
    pub view: Option<ViewState>,
    pub gpu: bool,
    pub textures: bool,
    pub blits: u64,
    pub cycles_notice: bool,
}

impl Live {
    pub fn info(&self) -> LiveInfo {
        LiveInfo {
            connected: self.session.connected(),
            closed: self.closed.clone(),
            age: self.session.age(),
            hello: self.hello.clone(),
            landed: self.landed,
            error: self.error.clone(),
            frames_in: self.frames_in,
            dropped: self.dropped,
            cached: self.cache.len(),
            log: self.log.clone(),
            view: self.view,
            gpu: self.gpu.is_some(),
            textures: self.surfaces.is_some(),
            blits: self.blits,
            cycles_notice: self.cycles_notice,
        }
    }
}

/// What the panel needs to know about the session, read before the node
/// borrow (the session lives on the app, the params on the node).
pub struct View {
    pub live: Option<LiveInfo>,
    /// What the node's layer holds right now.
    pub stamp: Option<BlenderStamp>,
    /// The Blender frame the playhead asks for.
    pub want: i64,
    /// The comp format the draw matches.
    pub format: (u32, u32),
    pub working: engine::color::Colorspace,
    /// Where the bridge script is written for a hand install.
    pub addon_dir: PathBuf,
}

/// The properties panel. Params are edited in place (undo and save like
/// any knob); session buttons come back as an `Action`.
pub fn panel(ui: &mut Ui, node: u64, p: &mut BlenderParams, v: &View) -> Option<Action> {
    let mut action = None;
    if v.live.as_ref().is_some_and(|l| l.cycles_notice) && cycles_notice(ui.ctx(), node) {
        action = Some(Action::DismissCyclesNotice);
    }
    // --- status ------------------------------------------------------
    let status = match &v.live {
        None => "no session".to_string(),
        Some(l) => {
            if let Some(why) = &l.closed {
                format!("closed: {why}")
            } else if !l.connected {
                format!("starting… {:.0}s", l.age)
            } else {
                match (&l.hello, v.stamp) {
                    (Some(h), Some(s)) => {
                        let what = if s.frame == v.want {
                            if s.settled {
                                if s.is_final { "final".to_string() } else { "settled".to_string() }
                            } else {
                                let n = l.landed.map(|x| x.samples).unwrap_or(0);
                                format!("refining, {n} samples")
                            }
                        } else {
                            format!("showing {}, asked {}", s.frame, v.want)
                        };
                        format!("blender {} — frame {} ({what})", h.version, s.frame)
                    }
                    (Some(h), None) => format!("blender {} — waiting for a frame", h.version),
                    (None, _) => "connected, waiting for hello".to_string(),
                }
            }
        }
    };
    let stale = matches!(v.stamp, Some(s) if s.frame != v.want || !s.settled);
    ui.horizontal(|ui| {
        let text = egui::RichText::new(&status).small();
        if v.live.as_ref().is_some_and(|l| l.error.is_some()) {
            ui.colored_label(crate::theme::error(), text);
        } else if stale {
            ui.colored_label(crate::theme::cost_slow(), text);
        } else {
            ui.label(text);
        }
    });
    if let Some(l) = &v.live {
        if let Some(e) = &l.error {
            ui.colored_label(crate::theme::error(), egui::RichText::new(e).small());
        }
    }
    // --- session buttons ---------------------------------------------
    ui.horizontal(|ui| {
        let up = v.live.as_ref().is_some_and(|l| l.closed.is_none());
        if !up {
            if ui.button("open in blender").on_hover_text(
                "launch Blender on this .blend (or a fresh scene) with the wobble bridge; \
                 the playhead drives it and its camera's picture comes back here",
            ).clicked() {
                action = Some(Action::Open);
            }
            let port_id = ui.id().with(("blender-port", node));
            let mut port: u32 = ui.ctx().data(|d| d.get_temp(port_id)).unwrap_or(47100);
            ui.label(egui::RichText::new("or port").small());
            let r = crate::widgets::num_field_w(ui, widgets::NUM_W + 12.0, egui::DragValue::new(&mut port).update_while_editing(false).range(1..=65535).speed(1.0),
            );
            if r.changed() {
                ui.ctx().data_mut(|d| d.insert_temp(port_id, port));
            }
            if ui.button("connect").on_hover_text(
                "join a Blender you already have open: install the wobble addon there \
                 (the addon file button writes it), set this port in its sidebar tab \
                 (View3D > sidebar > wobble) and press its connect, then this one",
            ).clicked() {
                action = Some(Action::Connect(port as u16));
            }
            if ui.button("addon file").on_hover_text(
                "write the Blender addon (wobble_blender.py) and copy its path: install it in \
                 Blender's preferences > add-ons > install from disk",
            ).clicked() {
                match blender::install_script(&v.addon_dir) {
                    Ok(path) => ui.ctx().copy_text(path.display().to_string()),
                    Err(e) => log::warn!("blender addon: {e}"),
                }
            }
        } else {
            if ui.button("refresh").on_hover_text("ask for the current frame again").clicked() {
                action = Some(Action::Refresh);
            }
            if ui.button("detach").on_hover_text("drop the link; Blender stays open").clicked() {
                action = Some(Action::Detach);
            }
            if ui.button("quit blender").on_hover_text("ask Blender to close (it asks about unsaved work)").clicked() {
                action = Some(Action::Quit);
            }
        }
    });
    // --- the scene -----------------------------------------------------
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new(".blend").small()));
        let r = ui.add(egui::TextEdit::singleline(&mut p.blend).desired_width(f32::INFINITY).hint_text("empty = a fresh scene; saving in Blender fills this in"));
        r.on_hover_text("~ expands; the file is opened as is, nothing in it is changed except the format, engine and camera below");
    });
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("scene").small()));
        ui.add(egui::TextEdit::singleline(&mut p.scene).desired_width(90.0).hint_text("active"));
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("camera").small()));
        ui.add(egui::TextEdit::singleline(&mut p.camera).desired_width(90.0).hint_text("scene camera"));
    });
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("picture").small()));
        egui::ComboBox::from_id_salt(("blender-quality", node))
            .selected_text(p.quality.name())
            .show_ui(ui, |ui| {
                for q in Quality::ALL {
                    ui.selectable_value(&mut p.quality, q, q.name());
                }
            })
            .response
            .on_hover_text(
                "viewport: the 3D view drawn through the camera, live, refined until it \
                 settles — SLOW (a round trip a frame). final: a real render of each \
                 frame, read back when it completes — HEAVY (seconds a frame in Cycles)",
            );
        if p.quality == Quality::Viewport {
            egui::ComboBox::from_id_salt(("blender-shading", node))
                .selected_text(p.shading.name())
                .show_ui(ui, |ui| {
                    for s in blender::Shading::ALL {
                        ui.selectable_value(&mut p.shading, s, s.name());
                    }
                })
                .response
                .on_hover_text("the viewport shading the draw uses; set on the 3D view, so Blender shows what wobble gets");
        }
        egui::ComboBox::from_id_salt(("blender-engine", node))
            .selected_text(p.engine.name())
            .show_ui(ui, |ui| {
                for e in blender::BlenderEngine::ALL {
                    ui.selectable_value(&mut p.engine, e, e.name());
                }
            });
    });
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("samples").small()));
        crate::widgets::num_field(ui, egui::DragValue::new(&mut p.samples).update_while_editing(false).range(0..=4096).speed(widgets::rated(ui, 0.5)),
        )
        .on_hover_text("viewport: draws to settle at (0 = the scene's viewport samples); final: the render's samples (0 = the file's)");
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("offset").small()));
        crate::widgets::num_field(ui, egui::DragValue::new(&mut p.offset).update_while_editing(false).speed(widgets::rated(ui, 1.0)),
        )
        .on_hover_text("comp frame = Blender frame + offset: a scene animated 1..48 with 1000 here plays at 1001..1048");
        ui.checkbox(&mut p.transparent, "transparent").on_hover_text("film > transparent: the world is alpha 0, straight into an over");
    });
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("inputs").small()));
        let mut n = p.inputs as u32;
        if ui
            .add_sized(
                [widgets::NUM_W, widgets::NUM_H],
                egui::DragValue::new(&mut n).update_while_editing(false).range(0..=blender::MAX_INPUTS as u32).speed(0.05),
            )
            .on_hover_text(
                "image ports: each arrives in Blender as the image wobble_in_N (scene-linear), \
                 for any material's Image Texture node. Exported again whenever the picture \
                 above the port changes — a render and a readback each time",
            )
            .changed()
        {
            p.inputs = n as usize;
        }
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("playhead").small()));
        egui::ComboBox::from_id_salt(("blender-sync", node))
            .selected_text(p.sync.name())
            .show_ui(ui, |ui| {
                for s in blender::Sync::ALL {
                    ui.selectable_value(&mut p.sync, s, s.name());
                }
            })
            .response
            .on_hover_text("which timeline follows which; both ways by default, and a move on one side never echoes");
    });
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("eye").small()));
        egui::ComboBox::from_id_salt(("blender-view", node))
            .selected_text(p.view.name())
            .show_ui(ui, |ui| {
                for m in blender::View::ALL {
                    ui.selectable_value(&mut p.view, m, m.name());
                }
            })
            .response
            .on_hover_text(
                "camera: the scene camera's picture. free: the 3D view's own eye — what \
                 Blender's viewport shows, orbitable from either side, not saved with the file",
            );
        ui.checkbox(&mut p.orbit, "orbit here").on_hover_text(
            "Blender's hands over the picture in the viewer move the eye: MMB orbit, \
             shift+MMB pan, ctrl+MMB or wheel dolly (the 2D pan stays on alt+LMB / \
             \u{2318}-drag). In camera mode it moves the camera — with autokey on in \
             Blender, an orbit becomes camera animation",
        );
        ui.checkbox(&mut p.cards, "cards").on_hover_text(
            "mount every input on a card in front of the camera: a plane carrying \
             wobble_in_N as unlit emission through its own alpha, fitted to the \
             frame, one unit further back per port",
        );
    });
    ui.horizontal_wrapped(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("passes").small()));
        ui.checkbox(&mut p.fat, "fat cable").on_hover_text(
            "output 0 carries the combined picture and every pass by name, like a multi-layer \
             Read: wire it into a Breakout or a ChannelSelect. The per-pass outputs stay",
        );
        for pass in blender::Pass::ALL {
            let on = p.passes.contains(&pass);
            let out = p.passes.iter().position(|x| *x == pass).map(|i| i + 1);
            let label = match out {
                Some(k) => format!("{} ·{k}", pass.name()),
                None => pass.name().to_string(),
            };
            if ui.selectable_label(on, egui::RichText::new(label).small()).on_hover_text(
                "an extra output of the node (in the order picked; output 0 is the combined picture): \
                 the viewport draws the pass on its own, one more draw a frame. Mist, AO and shadow \
                 ride the data pipe (a Defocus Z port takes mist as is). Viewport mode only",
            ).clicked() {
                if on {
                    p.passes.retain(|x| *x != pass);
                } else {
                    p.passes.push(pass);
                }
            }
        }
    });
    crate::colorui::input_space_row(ui, "blender-space", &mut p.space, v.working, "x.exr");
    // --- what is going on -------------------------------------------
    if let Some(l) = &v.live {
        let mut bits = Vec::new();
        if let Some(h) = &l.hello {
            bits.push(format!("scene {}..{} @ {}fps", h.start, h.end, h.fps));
        }
        if let Some(x) = l.landed {
            bits.push(format!("last frame {:.0}ms", x.ms));
        }
        bits.push(format!("{} frames in, {} cached", l.frames_in, l.cached));
        bits.push(match (l.textures, l.gpu) {
            (true, _) => format!("shared textures, no readback ({} blits)", l.blits),
            (false, true) => format!("ring on the GPU ({} blits)", l.blits),
            _ => "ring on the CPU path".into(),
        });
        if l.dropped > 0 {
            bits.push(format!("{} dropped", l.dropped));
        }
        if v.format != (0, 0) {
            bits.push(format!("{}x{}", v.format.0, v.format.1));
        }
        if let Some(e) = l.view {
            bits.push(format!("eye {:.1} {:.1} {:.1} @ {:.1}{}", e.loc[0], e.loc[1], e.loc[2], e.dist, if e.ortho { " ortho" } else { "" }));
        }
        ui.weak(bits.join(" · "));
        for line in &l.log {
            ui.label(egui::RichText::new(line).small().weak());
        }
    }
    ui.weak(match p.quality {
        Quality::Viewport => "SLOW PATH: every frame is a round trip to Blender. A Bake below this node turns it into files.",
        Quality::Final => "HEAVY: every frame is a full render in Blender. A Bake below this node turns it into files.",
    });
    action
}

/// The graph glyph: a camera — a body and a lens, the picture comes from
/// somewhere else.
pub fn glyph(painter: &egui::Painter, c: egui::Pos2, r: f32, fill: egui::Color32, stroke: egui::Stroke) {
    let body = egui::Rect::from_center_size(egui::pos2(c.x - r * 0.15, c.y), egui::vec2(r * 1.5, r * 1.1));
    painter.rect(body, r * 0.2, fill, stroke, egui::StrokeKind::Middle);
    let lens = [
        egui::pos2(c.x + r * 0.55, c.y - r * 0.3),
        egui::pos2(c.x + r * 1.05, c.y - r * 0.6),
        egui::pos2(c.x + r * 1.05, c.y + r * 0.6),
        egui::pos2(c.x + r * 0.55, c.y + r * 0.3),
    ];
    painter.add(egui::Shape::convex_polygon(lens.to_vec(), fill, stroke));
    painter.circle_stroke(egui::pos2(c.x - r * 0.15, c.y), r * 0.3, stroke);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blender_5_1_is_called_out_and_its_neighbours_are_not() {
        assert!(broken_release("5.1.0").is_some());
        assert!(broken_release("5.1.2").is_some());
        assert!(broken_release("5.2.1 LTS").is_none());
        assert!(broken_release("5.10.0").is_none());
        assert!(broken_release("4.2.3 LTS").is_none());
        assert!(broken_release("").is_none());
    }

    fn frame(f: i64, scene: u64, settled: bool) -> shm::Frame {
        shm::Frame {
            seq: 2,
            frame: f,
            w: 2,
            h: 1,
            fmt: shm::FMT_RGBA16F,
            flags: if settled { shm::FLAG_SETTLED } else { 0 },
            scene,
            ms: 1.0,
            samples: 1,
            pass: 0,
            pixels: vec![f as u8; 16],
        }
    }

    /// A session that never connects: enough to drive the bookkeeping.
    fn idle() -> Live {
        let port = live::free_port().unwrap();
        Live::new(live::Session::connect(port), engine::paths::temp_dir())
    }

    #[test]
    fn settled_frames_are_cached_per_scene_version_and_an_edit_empties_it() {
        let mut l = idle();
        l.scene = 3;
        l.remember(10, 0, 2, 1, shm::FMT_RGBA16F, false, Arc::new(frame(10, 3, true).pixels));
        l.remember(11, 0, 2, 1, shm::FMT_RGBA16F, false, Arc::new(frame(11, 3, true).pixels));
        assert!(l.has_cached(10) && l.has_cached(11));
        // The version moves: everything cached was drawn for an old scene.
        l.scene = 4;
        l.cache.clear();
        l.cache_bytes = 0;
        assert!(!l.has_cached(10));
    }

    #[test]
    fn the_cache_evicts_the_farthest_frame_first() {
        let mut l = idle();
        // Two fit, three do not.
        let big = vec![0u8; CACHE_BYTES / 3 + 1];
        l.remember(1, 0, 1, 1, shm::FMT_RGBA16F, false, Arc::new(big.clone()));
        l.remember(50, 0, 1, 1, shm::FMT_RGBA16F, false, Arc::new(big.clone()));
        assert!(l.has_cached(1) && l.has_cached(50));
        // Adding 52 evicts 1 (farthest from 52), keeps 50.
        l.remember(52, 0, 1, 1, shm::FMT_RGBA16F, false, Arc::new(big));
        assert!(!l.has_cached(1));
        assert!(l.has_cached(50) && l.has_cached(52));
        assert!(l.cache_bytes <= CACHE_BYTES);
    }

    #[test]
    fn a_sync_mode_says_who_follows() {
        use engine::blender::Sync;
        assert!(Sync::Both.drives_blender() && Sync::Both.follows_blender());
        assert!(Sync::Wobble.drives_blender() && !Sync::Wobble.follows_blender());
        assert!(!Sync::Blender.drives_blender() && Sync::Blender.follows_blender());
    }

    /// The panel must FIT the tiles it asks for, at its fullest: a live
    /// session with a hello, a landed frame, an error and a full log. A
    /// row pushed into the scroll area cannot be clicked (the ColorCompress
    /// precedent), so this measures rather than trusts.
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let mut p = BlenderParams::default();
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        let span = |n: usize| {
            crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
        };
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        let view = View {
            live: Some(LiveInfo {
                connected: true,
                closed: None,
                age: 3.0,
                hello: Some(Hello {
                    version: "5.2.1 LTS".into(),
                    w: 2048,
                    h: 1080,
                    fps: 24.0,
                    blend: "/x/y.blend".into(),
                    frame: 1,
                    start: 1,
                    end: 48,
                }),
                landed: Some(Landed {
                    frame: 7,
                    scene: 2,
                    settled: false,
                    is_final: false,
                    ms: 12.5,
                    samples: 16,
                    w: 2048,
                    h: 1080,
                }),
                error: Some("draw: no 3D viewport to draw from".into()),
                frames_in: 40,
                dropped: 1,
                cached: 12,
                log: vec!["a".into(), "b".into(), "c".into()],
                view: Some(ViewState::default()),
                gpu: true,
                textures: true,
                blits: 40,
                cycles_notice: false,
            }),
            stamp: Some(BlenderStamp { frame: 6, scene: 2, settled: true, is_final: false }),
            want: 7,
            format: (2048, 1080),
            working: engine::color::Colorspace::ACESCG,
            addon_dir: engine::paths::temp_dir(),
        };
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let _ = panel(ui, 1, &mut p, &view);
                });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("blender-panel");
        assert!(over.get() <= 0.5, "the panel overflows its {TILE_ROWS} rows by {}pt", over.get());
        assert!(wide.get() <= 0.5, "the panel overflows its {TILES} tiles sideways by {}pt", wide.get());
    }
}

// --------------------------------------------------------------------------
// The app side: what a redraw does with every session.

use crate::Gfx;
use engine::graph::NodeKind;

impl Gfx {
    /// Where the sessions' files live: the script and the pixel rings.
    /// Always the system temp dir, never the frame cache's: on Windows a
    /// ring is a file mapping, and a cache pointed at a small or full drive
    /// (a nearly full E: on one machine) failed every ring write with ENOSPC, which froze the
    /// picture the moment the live path fell back to the ring.
    pub(crate) fn blender_dir(&self) -> PathBuf {
        engine::paths::temp_dir().join("wobble-blender")
    }

    /// The tick every Blender node should be showing: a running bake's
    /// next frame, else where the user is (the hand's target, not the
    /// neighbour a scrub happens to have on screen).
    pub(crate) fn blender_wanted_tick(&self) -> u64 {
        if let Some(b) = &self.graph_bake {
            b.next
        } else if let Some(b) = &self.render_bake {
            b.next
        } else {
            self.seek_target.unwrap_or(self.timeline_pos)
        }
    }

    /// (node, scene version) for every session: part of the bake keys, so
    /// an edit made in Blender is a different bake.
    pub(crate) fn blender_versions(&self) -> Vec<(u64, u64)> {
        let mut v: Vec<(u64, u64)> = self.blender.iter().map(|(id, l)| (*id, l.scene)).collect();
        v.sort_unstable();
        v
    }

    fn blender_spec(&self, p: &BlenderParams) -> String {
        let (w, h) = self.comp.output_size().unwrap_or((0, 0));
        p.spec_line(w, h)
    }

    /// Once a redraw, before the bakes step and the display loop encodes:
    /// drain every session, put what landed on the canvas, send the
    /// playhead, follow Blender's.
    pub(crate) fn blender_step(&mut self) {
        if self.blender.is_empty() {
            return;
        }
        // A deleted node's session is dropped; Blender keeps running.
        let egraph = &self.egraph;
        self.blender.retain(|id, _| {
            matches!(egraph.node(*id).map(|n| &n.kind), Some(NodeKind::Blender(_)))
        });
        let baking = self.graph_bake.is_some() || self.render_bake.is_some();
        let wanted = self.blender_wanted_tick();
        let ids: Vec<u64> = self.blender.keys().copied().collect();
        let mut follow: Option<u64> = None;
        let mut landed_any = false;
        let mut learned: Vec<(u64, String)> = Vec::new();
        // Blender nodes a 3D Camera or Card reads its pose from.
        let scene3d: std::collections::HashSet<u64> = self
            .egraph
            .nodes
            .values()
            .filter_map(|n| match &n.kind {
                NodeKind::Camera3D(engine::graph::CameraParams { source: engine::scene3d::Source::Blender { node, .. }, .. })
                | NodeKind::Card3D(engine::graph::CardParams { source: engine::scene3d::Source::Blender { node, .. }, .. }) => {
                    Some(*node)
                }
                _ => None,
            })
            .collect();
        for id in ids {
            let Some(NodeKind::Blender(p)) = self.egraph.node(id).map(|n| n.kind.clone()) else {
                continue;
            };
            let spec = self.blender_spec(&p);
            let live = self.blender.get_mut(&id).unwrap();
            let before = live.hello.as_ref().map(|h| h.blend.clone());
            let (landings, blits) = live.poll(&p, &spec, &self.ctx.device);
            for Landing { pass, w, h, fmt, pixels, stamp } in landings {
                self.comp.blender_upload(id, pass, w, h, fmt, &pixels, stamp);
                landed_any = true;
            }
            for b in blits {
                let st = b.stamp_meta;
                let ok = match &b.source {
                    BlitSource::Ring { ring, offset, stride } => self.comp.blender_copy(
                        id, b.pass, &ring.buffer, *offset, *stride, st.w, st.h, st.fmt, b.stamp, b.done,
                    ),
                    BlitSource::Surface(surfaces) => self.comp.blender_copy_texture(
                        id, b.pass, &surfaces[b.slot].texture, st.w, st.h, st.fmt, b.stamp, b.done,
                    ),
                };
                if ok {
                    landed_any = true;
                }
            }
            // Live Cycles (a wobble-fork Blender): a completed display
            // update sits in the shared buffer. Rows into the staging
            // buffer's 256-byte stride, then the ring's GPU path.
            if let Some(t) = live.cycles.as_mut() {
                if let Some(u) = t.poll() {
                    live.cycles_updates += 1;
                    let mut enc = self
                        .ctx
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("cycles-rows") });
                    t.encode_rows(&mut enc, &u);
                    self.ctx.queue.submit([enc.finish()]);
                    let stamp = BlenderStamp {
                        frame: live.sent.unwrap_or(0),
                        scene: live.scene,
                        settled: false,
                        is_final: false,
                    };
                    let stride = blender::cycles::padded_stride(u.w);
                    if self.comp.blender_copy(
                        id, 0, &t.staging, 0, stride, u.w, u.h, shm::FMT_RGBA16F, stamp, Box::new(|| {}),
                    ) {
                        landed_any = true;
                    }
                }
            }
            // A fresh scene saved in Blender: the node learns its file.
            if p.blend.trim().is_empty() {
                if let Some(saved) = live.log.iter().rev().find_map(|l| l.strip_prefix("saved ")) {
                    if !saved.is_empty() && before.as_deref() != Some(saved) {
                        learned.push((id, saved.to_string()));
                    }
                }
            }
            if let Some(f) = live.follow.take() {
                follow = Some(p.tick_of(f));
            }
            // 3D cards: ask for the scene's cameras and planes whenever its
            // version moves (an edit in Blender bumps it), and point the
            // cards at the file once it is written.
            if scene3d.contains(&id) && live.hello.is_some() && live.scene3d_asked != Some(live.scene) {
                let path = live.dir.join("scene3d.json");
                live.session.scene3d(&path.to_string_lossy());
                live.scene3d_asked = Some(live.scene);
            }
            if let Some(f) = &live.scene3d_file {
                engine::scene3d::set_blender_scene(id, f.clone(), -(p.offset as f32));
                landed_any |= std::mem::take(&mut live.scene3d_fresh);
            }
            let frame = p.frame_of(wanted);
            let stamp = self.comp.blender_stamp(id);
            let current =
                stamp.is_some_and(|s| s.frame == frame && s.scene == live.scene && s.settled);
            let mut cached = current;
            if !current && live.has_cached(frame) {
                for Landing { pass, w, h, fmt, pixels, stamp } in live.cached(frame) {
                    self.comp.blender_upload(id, pass, w, h, fmt, &pixels, stamp);
                    landed_any = true;
                }
                cached = true;
            }
            live.drive(&p, &spec, frame, cached);
        }
        for (id, path) in learned {
            if let Some(NodeKind::Blender(p)) = self.egraph.node_mut(id).map(|n| &mut n.kind) {
                if p.blend.trim().is_empty() {
                    p.blend = path;
                }
            }
        }
        if landed_any {
            // A layer changed under an unchanged graph: the change gate
            // would idle-skip the render and show the old picture.
            self.last_engine_graph = None;
        }
        if let Some(t) = follow {
            if !baking && t != self.seek_target.unwrap_or(self.timeline_pos) {
                self.seek_to(t);
            }
        }
    }

    /// At the end of a redraw, with this tick's plates in the reader
    /// textures: hand Blender every input port whose picture changed. A
    /// full render and a readback per port — the slow path this node is
    /// red for.
    pub(crate) fn blender_export(&mut self) {
        if self.blender.is_empty() {
            return;
        }
        let tick = self.blender_wanted_tick();
        let ids: Vec<u64> = self.blender.keys().copied().collect();
        let mut fmts: Option<Vec<(u64, (u32, u32))>> = None;
        let mut exported = false;
        for id in ids {
            let Some(NodeKind::Blender(p)) = self.egraph.node(id).map(|n| n.kind.clone()) else {
                continue;
            };
            if !self.blender.get(&id).is_some_and(|l| l.up()) {
                continue;
            }
            for port in 0..p.input_count() {
                let Some(src) = self.egraph.input_of(id, port) else { continue };
                let Some(key) = self.subtree_key_of(src) else { continue };
                if self.blender[&id].tex_sent.get(&port) == Some(&(key, tick)) {
                    continue;
                }
                let fmts = fmts.get_or_insert_with(|| self.collect_formats());
                let mut g = self.egraph.to_engine_graph_at(Some(tick));
                g.formats.extend(fmts.iter().copied());
                self.apply_bake_state(&mut g);
                g.view = Some(src);
                if !self.comp.set_graph(&g) {
                    continue;
                }
                self.comp.write_params(&g);
                let mut enc = self.ctx.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("blender-export") },
                );
                let samples =
                    crate::motion_blur_samples(&self.egraph, &self.comp, tick, &g, |tf| {
                        self.egraph.to_engine_graph_at_f(Some(tf))
                    });
                crate::encode_motion_blurs(
                    &mut self.comp,
                    &self.ctx,
                    &self.timer,
                    &mut enc,
                    samples,
                    &g,
                );
                self.comp.encode_comp(&mut enc, &self.timer);
                if self.comp.params_stale() {
                    self.comp.write_params(&g);
                }
                self.ctx.queue.submit([enc.finish()]);
                exported = true;
                let Some((px, w, h)) = self.comp.read_output_rgba16() else { continue };
                let live = self.blender.get_mut(&id).unwrap();
                live.send_texture(port, w, h, &px);
                live.tex_sent.insert(port, (key, tick));
            }
        }
        if exported {
            // The compositor now holds the export's graph and params; the
            // next redraw must set and pack the comp's own again.
            self.last_engine_graph = None;
        }
    }

    /// Every Blender node upstream of `root` shows the settled frame for
    /// `tick` — or has no live session to wait for. What a bake waits on
    /// before it writes a frame.
    pub(crate) fn blender_ready(&self, root: u64, tick: u64) -> bool {
        let mut stack = vec![root];
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let Some(n) = self.egraph.node(id) else { continue };
            if let NodeKind::Blender(p) = &n.kind {
                if let Some(l) = self.blender.get(&id) {
                    if l.up() {
                        let ok = self.comp.blender_stamp(id).is_some_and(|s| {
                            s.frame == p.frame_of(tick) && s.settled && s.scene == l.scene
                        });
                        if !ok {
                            return false;
                        }
                    }
                }
            }
            for i in 0..n.kind.input_count() {
                if let Some(s) = self.egraph.input_of(id, i) {
                    stack.push(s);
                }
            }
        }
        true
    }

    /// The Blender node whose eye the viewer's MMB chords move: the view
    /// target, with `orbit` on and a session up. `None` = the chords pan
    /// the 2D picture as everywhere else.
    pub(crate) fn blender_nav_target(&self) -> Option<u64> {
        let id = self.egraph.active_view()?;
        match self.egraph.node(id).map(|n| &n.kind) {
            Some(NodeKind::Blender(p)) if p.orbit => {}
            _ => return None,
        }
        self.blender.get(&id).filter(|l| l.up()).map(|_| id)
    }

    /// One step of the hands, from the viewer: `px_high` is how tall the
    /// picture is on screen, so a pan follows the pointer at the pivot.
    pub(crate) fn blender_nav(&mut self, id: u64, g: Nav, px_high: f32) {
        let Some(NodeKind::Blender(p)) = self.egraph.node(id).map(|n| n.kind.clone()) else {
            return;
        };
        let format = self.comp.output_size().unwrap_or((0, 0));
        if let Some(l) = self.blender.get_mut(&id) {
            l.nav(&p, g, format, px_high);
        }
    }

    /// A two-finger / wheel step on the eye (see `Live::nav_pad`).
    pub(crate) fn blender_nav_pad(&mut self, id: u64, g: Nav, px_high: f32) {
        let Some(NodeKind::Blender(p)) = self.egraph.node(id).map(|n| n.kind.clone()) else {
            return;
        };
        let format = self.comp.output_size().unwrap_or((0, 0));
        if let Some(l) = self.blender.get_mut(&id) {
            l.nav_pad(&p, g, format, px_high);
        }
    }

    /// End two-finger gestures that have come to rest; true while one is open.
    pub(crate) fn blender_pad_idle(&mut self) -> bool {
        let mut open = false;
        for l in self.blender.values_mut() {
            open |= l.pad_idle();
        }
        open
    }

    pub(crate) fn blender_nav_end(&mut self, id: u64) {
        if let Some(l) = self.blender.get_mut(&id) {
            l.nav_end();
        }
    }

    /// `blender_status` over the API: the panel's facts as JSON.
    pub(crate) fn blender_status_json(&self, id: u64) -> serde_json::Value {
        use serde_json::json;
        let want = match self.egraph.node(id).map(|n| &n.kind) {
            Some(NodeKind::Blender(p)) => Some(p.frame_of(self.blender_wanted_tick())),
            _ => None,
        };
        let stamp = self.comp.blender_stamp(id).map(|s| {
            json!({ "frame": s.frame, "scene": s.scene, "settled": s.settled, "final": s.is_final })
        });
        let passes: Vec<serde_json::Value> = match self.egraph.node(id).map(|n| &n.kind) {
            Some(NodeKind::Blender(p)) => p
                .passes
                .iter()
                .enumerate()
                .map(|(k, pass)| {
                    let s = self.comp.blender_pass_stamp(id, pass.id());
                    json!({
                        "output": k + 1,
                        "pass": pass.name(),
                        "id": pass.id(),
                        "canvas": s.map(|s| json!({ "frame": s.frame, "scene": s.scene, "settled": s.settled })),
                    })
                })
                .collect(),
            _ => Vec::new(),
        };
        let Some(l) = self.blender.get(&id) else {
            return json!({ "session": false, "want": want, "canvas": stamp, "passes": passes });
        };
        let i = l.info();
        json!({
            "session": true,
            "connected": i.connected,
            "closed": i.closed,
            "age": i.age,
            "port": l.session.port,
            "blender": i.hello.as_ref().map(|h| json!({
                "version": h.version, "w": h.w, "h": h.h, "fps": h.fps, "blend": h.blend,
                "frame": h.frame, "start": h.start, "end": h.end,
            })),
            "want": want,
            "canvas": stamp,
            "passes": passes,
            "current": stamp.as_ref().zip(want).is_some_and(|(s, w)| s["frame"] == json!(w) && s["settled"] == json!(true) && s["scene"] == json!(l.scene)),
            "landed": i.landed.map(|x| json!({
                "frame": x.frame, "scene": x.scene, "settled": x.settled, "final": x.is_final,
                "ms": x.ms, "samples": x.samples, "w": x.w, "h": x.h,
            })),
            "scene": l.scene,
            "eye": i.view.map(|v| json!({ "loc": v.loc, "rot": v.rot, "dist": v.dist, "lens": v.lens, "ortho": v.ortho })),
            "frames_in": i.frames_in,
            "dropped": i.dropped,
            "cached": i.cached,
            "error": i.error,
            "log": i.log,
        })
    }

    /// The snapshot the panel draws from, taken before the node borrow.
    pub(crate) fn blender_view(&self, node: u64) -> View {
        let want = match self.egraph.node(node).map(|n| &n.kind) {
            Some(NodeKind::Blender(p)) => p.frame_of(self.blender_wanted_tick()),
            _ => 0,
        };
        View {
            live: self.blender.get(&node).map(|l| l.info()),
            stamp: self.comp.blender_stamp(node),
            want,
            format: self.comp.output_size().unwrap_or((0, 0)),
            working: self.egraph.working,
            addon_dir: self.blender_dir(),
        }
    }

    pub(crate) fn blender_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Open => self.blender_open(node),
            Action::Connect(port) => {
                let dir = self.blender_dir();
                let _ = std::fs::create_dir_all(&dir);
                self.blender.insert(node, Live::new(live::Session::connect(port), dir));
                self.load_errors.remove(&node);
            }
            Action::Detach => {
                self.blender.remove(&node);
            }
            Action::Quit => {
                if let Some(l) = self.blender.remove(&node) {
                    l.session.quit();
                }
            }
            Action::DismissCyclesNotice => {
                if let Some(l) = self.blender.get_mut(&node) {
                    l.cycles_notice = false;
                }
            }
            Action::Refresh => {
                if let Some(l) = self.blender.get_mut(&node) {
                    l.sent = None;
                    l.spec_sent = None;
                    l.tex_sent.clear();
                }
            }
        }
    }

    /// Launch Blender on the node's `.blend` with the bridge, listening on
    /// a port of our choosing.
    fn blender_open(&mut self, node: u64) {
        let Some(NodeKind::Blender(p)) = self.egraph.node(node).map(|n| n.kind.clone()) else {
            return;
        };
        let dir = self.blender_dir();
        let script = match blender::install_script(&dir) {
            Ok(s) => s,
            Err(e) => {
                self.load_errors.insert(node, format!("blender bridge: {e}"));
                return;
            }
        };
        let port = match live::free_port() {
            Ok(p) => p,
            Err(e) => {
                self.load_errors.insert(node, format!("blender: {e:#}"));
                return;
            }
        };
        let blend = p.blend_path();
        match live::Session::launch(&blender::executable(), blend.as_deref(), &script, port, &dir) {
            Ok(s) => {
                self.blender.insert(node, Live::new(s, dir));
                self.load_errors.remove(&node);
            }
            Err(e) => {
                self.load_errors.insert(node, format!("{e:#}"));
            }
        }
    }
}

// --------------------------------------------------------------------------
// A stand-in Blender for `--test-blender`.

/// Speaks the bridge protocol on a real socket and a real ring, "draws" a
/// flat colour that encodes the frame and the scene version (r = frame /
/// 100, g = scene / 10, b = 0.5), and records what wobble asked for — so
/// the suite can check the playhead, the cache and the scene edit without
/// a Blender.
pub mod fake {
    use engine::blender::shm;
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    pub struct State {
        /// Every `spec` line received.
        pub specs: Mutex<Vec<String>>,
        /// Every `frame` request: (frame, cached).
        pub frames: Mutex<Vec<(i64, bool)>>,
        pub scene: AtomicU64,
        /// `scene3d` requests answered (the 3D cards' export).
        pub scene3d_asks: AtomicU64,
        current: AtomicU64,
        conn: Mutex<Option<TcpStream>>,
        ring: Mutex<Option<shm::Writer>>,
        w: u32,
        h: u32,
    }

    impl State {
        fn send(&self, line: String) {
            if let Some(c) = self.conn.lock().unwrap().as_mut() {
                let _ = c.write_all(line.as_bytes());
            }
        }

        fn pixels(&self, frame: i64) -> Vec<u8> {
            let scene = self.scene.load(Ordering::Relaxed);
            let px = [
                half::f16::from_f32(frame as f32 / 100.0),
                half::f16::from_f32(scene as f32 / 10.0),
                half::f16::from_f32(0.5),
                half::f16::ONE,
            ];
            let mut out = Vec::with_capacity(self.w as usize * self.h as usize * 8);
            for _ in 0..self.w * self.h {
                out.extend_from_slice(bytemuck::cast_slice(&px));
            }
            out
        }

        fn draw(&self, frame: i64) {
            let px = self.pixels(frame);
            let stamp = shm::Stamp {
                frame,
                w: self.w,
                h: self.h,
                fmt: shm::FMT_RGBA16F,
                flags: shm::FLAG_SETTLED,
                scene: self.scene.load(Ordering::Relaxed),
                ms: 1.0,
                samples: 1,
                pass: 0,
            };
            let mut ring = self.ring.lock().unwrap();
            if let Some(r) = ring.as_mut() {
                if let Ok((slot, seq)) = r.write(stamp, &px) {
                    drop(ring);
                    self.send(format!("{{\"ev\": \"frame\", \"slot\": {slot}, \"seq\": {seq}}}\n"));
                }
            }
        }

        /// The user scrubbed in Blender.
        pub fn frame_changed(&self, frame: i64) {
            self.send(format!("{{\"ev\": \"frame_changed\", \"frame\": {frame}}}\n"));
        }

        /// The user edited the scene: a new version, and the current frame
        /// is redrawn with it.
        pub fn edit(&self) {
            let v = self.scene.fetch_add(1, Ordering::Relaxed) + 1;
            self.send(format!("{{\"ev\": \"scene\", \"scene\": {v}}}\n"));
            let cur = self.current.load(Ordering::Relaxed) as i64;
            self.draw(cur);
        }

        pub fn last_frame_request(&self) -> Option<(i64, bool)> {
            self.frames.lock().unwrap().last().copied()
        }
    }

    /// Listen on a free port; returns it and the shared state.
    pub fn start(dir: std::path::PathBuf, w: u32, h: u32) -> (u16, Arc<State>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let state = Arc::new(State {
            specs: Mutex::new(Vec::new()),
            frames: Mutex::new(Vec::new()),
            scene: AtomicU64::new(1),
            scene3d_asks: AtomicU64::new(0),
            current: AtomicU64::new(0),
            conn: Mutex::new(None),
            ring: Mutex::new(None),
            w,
            h,
        });
        let st = state.clone();
        std::thread::spawn(move || {
            let Ok((conn, _)) = listener.accept() else { return };
            let _ = conn.set_nodelay(true);
            let reader = BufReader::new(conn.try_clone().unwrap());
            *st.conn.lock().unwrap() = Some(conn);
            std::fs::create_dir_all(&dir).unwrap();
            let src = shm::Source::for_writer(&format!("wobfake_{port}"), &dir, 0);
            let ring = shm::Writer::create(&src, 3, shm::Writer::bytes_for(w, h, shm::FMT_RGBA16F).unwrap()).unwrap();
            let fields = ring.source.json_fields();
            *st.ring.lock().unwrap() = Some(ring);
            st.send(format!(
                "{{\"ev\": \"hello\", \"version\": \"fake\", \"w\": {w}, \"h\": {h}, \"fps\": 24, \"blend\": \"\", \"frame\": 1, \"start\": 1, \"end\": 48}}\n"
            ));
            st.send(format!("{{\"ev\": \"shm\", {fields}}}\n"));
            for line in reader.lines().map_while(Result::ok) {
                let Some(m) = engine::blender::live::parse_flat(&line) else { continue };
                let cmd = m.get("cmd").and_then(|v| match v {
                    engine::blender::live::Val::Str(s) => Some(s.as_str()),
                    _ => None,
                });
                match cmd {
                    Some("spec") => st.specs.lock().unwrap().push(line.clone()),
                    Some("frame") => {
                        let f = match m.get("frame") {
                            Some(engine::blender::live::Val::Num(n)) => *n as i64,
                            _ => continue,
                        };
                        let cached = matches!(m.get("cached"), Some(engine::blender::live::Val::Bool(true)));
                        st.frames.lock().unwrap().push((f, cached));
                        st.current.store(f as u64, Ordering::Relaxed);
                        if !cached {
                            st.draw(f);
                        }
                    }
                    // The 3D cards' export: a camera whose x is the scene
                    // version (so an edit moves it), five up looking down
                    // -Z at a two-unit plane on the origin. Y up.
                    Some("scene3d") => {
                        let Some(engine::blender::live::Val::Str(path)) = m.get("path") else { continue };
                        let v = st.scene.load(Ordering::Relaxed);
                        let json = format!(
                            r#"{{"objects": [{{"name": "Camera", "type": "camera", "frames": [{{"frame": 1, "translate": [{v}, 0, 5], "focal": 50, "aperture": [36, 24]}}]}},
                                {{"name": "Plane", "frames": [{{"frame": 1, "scale": [2, 2, 2]}}]}}]}}"#
                        );
                        if std::fs::write(path, json).is_ok() {
                            st.scene3d_asks.fetch_add(1, Ordering::Relaxed);
                            st.send(format!("{{\"ev\": \"scene3d\", \"path\": {}}}\n", engine::blender::live::json_str(path)));
                        }
                    }
                    Some("quit") => break,
                    _ => {}
                }
            }
            st.ring.lock().unwrap().take();
        });
        (port, state)
    }
}
