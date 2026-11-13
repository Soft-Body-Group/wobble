//! Wobble in a browser — phase 1: the ENGINE on WebGPU.
//!
//! The same `Compositor` the app drives (graph -> fused WGSL -> compute
//! passes -> blit) renders into a `<canvas>`, fed a procedural plate and
//! driven by the page's sliders through [`set`]. No app UI yet: that is
//! phase 2, and it is scoped in README.md.
//!
//! Every item is wasm32-only (the crate compiles empty on native, so the
//! workspace build and CI never need a web toolchain). What the browser
//! changes about the engine is gated at the source, not worked around here:
//! `GpuContext::new_async` (a page cannot block on the adapter),
//! `compile_plan` validating through naga instead of waiting on an error
//! scope, and `web_time::Instant` (std's panics on wasm32).
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;

use engine::blit::Blit;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{BlurParams, GlowParams, GradeParams, Graph, NodeKind, TransformParams};
use engine::timing::GpuTimer;
use engine::viewtf::ViewSettings;
use half::f16;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// The plate's size. 1080p is what a laptop GPU should chew through at 60Hz
/// with a blur and a glow in the chain, which is the claim the page makes.
const W: u32 = 1920;
const H: u32 = 1080;

// Node ids of the demo chain: Read -> Grade -> Transform -> Blur -> Glow.
const READ: u64 = 1;
const GRADE: u64 = 2;
const XFORM: u64 = 3;
const BLUR: u64 = 4;
const GLOW: u64 = 5;

struct Demo {
    ctx: engine::GpuContext,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    comp: Compositor,
    blit: Blit,
    view: ViewSettings,
    timer: GpuTimer,
    graph: Graph,
    /// Degrees the Transform sways either side: an animated param, which
    /// is a uniform write per frame and never a recompile.
    sway: f32,
    /// Params changed since the last `write_params`.
    dirty: bool,
    /// The topology hash the compositor is showing.
    live: Option<u64>,
    /// The last topology compile, for the stats line.
    compile_ms: f32,
    compiles: u32,
    /// The time [`frame`] was last called with, for the animation.
    t0: Option<f64>,
}

thread_local! {
    static DEMO: RefCell<Option<Demo>> = const { RefCell::new(None) };
}

/// `log` -> the browser console, so the engine's own logging reaches devtools.
struct Console;

impl log::Log for Console {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, r: &log::Record) {
        if !self.enabled(r.metadata()) {
            return;
        }
        let msg = JsValue::from_str(&format!("[{}] {}", r.target(), r.args()));
        match r.level() {
            log::Level::Error => web_sys::console::error_1(&msg),
            log::Level::Warn => web_sys::console::warn_1(&msg),
            _ => web_sys::console::log_1(&msg),
        }
    }
    fn flush(&self) {}
}

static LOGGER: Console = Console;

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
    let _ = log::set_logger(&LOGGER).map(|()| log::set_max_level(log::LevelFilter::Info));
}

/// Bring up WebGPU on the canvas with this id and compile the demo comp.
/// Resolves to a one-line description of the adapter; rejects with a
/// sentence a person can read (no WebGPU, no adapter, a failed compile).
#[wasm_bindgen]
pub async fn init(canvas_id: String) -> Result<String, JsError> {
    let canvas = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id(&canvas_id))
        .ok_or_else(|| JsError::new(&format!("no element #{canvas_id}")))?
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| JsError::new(&format!("#{canvas_id} is not a canvas")))?;
    let (cw, ch) = (canvas.width().max(1), canvas.height().max(1));

    let ctx = engine::GpuContext::new_async()
        .await
        .map_err(|e| JsError::new(&format!("WebGPU is not available here: {e}")))?;
    let info = ctx.adapter.get_info();
    let surface = ctx
        .instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(|e| JsError::new(&format!("canvas surface: {e}")))?;
    let caps = surface.get_capabilities(&ctx.adapter);
    // An 8-bit UNORM target: the blit applies the display encode itself.
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
        .or_else(|| caps.formats.first().copied())
        .ok_or_else(|| JsError::new("the canvas offers no surface format"))?;
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: cw,
        height: ch,
        present_mode: wgpu::PresentMode::Fifo,
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Opaque),
        view_formats: vec![],
    };
    surface.configure(&ctx.device, &config);

    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&plate()),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    comp.ensure_output(W, H);

    let view = ViewSettings::default();
    let mut blit = Blit::new(&ctx.device, format);
    blit.set_view(&ctx.device, &view);
    let timer = GpuTimer::new(&ctx);

    let mut demo = Demo {
        surface,
        config,
        comp,
        blit,
        view,
        timer,
        graph: demo_graph(),
        sway: 4.0,
        dirty: true,
        live: None,
        compile_ms: 0.0,
        compiles: 0,
        t0: None,
        ctx,
    };
    demo.compile()?;
    // Browsers usually withhold the adapter's name.
    let name = if info.name.is_empty() { "WebGPU" } else { info.name.as_str() };
    let line = format!("{name}, plate {W}x{H}, canvas {}", format_name(format));
    DEMO.with(|d| *d.borrow_mut() = Some(demo));
    Ok(line)
}

/// Move one control. Params are uniforms (no recompile); the `bypass_*`
/// switches change topology and recompile, which the stats line counts.
#[wasm_bindgen]
pub fn set(name: &str, value: f32) {
    DEMO.with(|d| {
        let mut d = d.borrow_mut();
        let Some(demo) = d.as_mut() else { return };
        let g = &mut demo.graph;
        match name {
            "gain" => {
                if let Some(NodeKind::Grade(p)) = g.nodes.get_mut(&GRADE) {
                    p.gain = [value; 3];
                }
            }
            "saturation" => {
                if let Some(NodeKind::Grade(p)) = g.nodes.get_mut(&GRADE) {
                    p.saturation = value;
                }
            }
            "blur" => {
                if let Some(NodeKind::Blur(p)) = g.nodes.get_mut(&BLUR) {
                    p.size = value;
                }
            }
            "glow" => {
                if let Some(NodeKind::Glow(p)) = g.nodes.get_mut(&GLOW) {
                    p.intensity = value;
                }
            }
            "threshold" => {
                if let Some(NodeKind::Glow(p)) = g.nodes.get_mut(&GLOW) {
                    p.threshold = value;
                }
            }
            "sway" => demo.sway = value,
            "exposure" => demo.view.exposure = value,
            _ => {
                let node = match name {
                    "bypass_grade" => GRADE,
                    "bypass_transform" => XFORM,
                    "bypass_blur" => BLUR,
                    "bypass_glow" => GLOW,
                    _ => {
                        log::warn!("unknown control {name:?}");
                        return;
                    }
                };
                if value > 0.5 {
                    g.disabled.insert(node);
                } else {
                    g.disabled.remove(&node);
                }
                if let Err(e) = demo.compile() {
                    log::error!("{e:?}");
                }
            }
        }
        demo.dirty = true;
    });
}

/// The canvas's backing store changed size (CSS size x devicePixelRatio).
#[wasm_bindgen]
pub fn resize(width: u32, height: u32) {
    DEMO.with(|d| {
        let mut d = d.borrow_mut();
        let Some(demo) = d.as_mut() else { return };
        demo.config.width = width.max(1);
        demo.config.height = height.max(1);
        demo.surface.configure(&demo.ctx.device, &demo.config);
    });
}

/// Render one frame. `now_ms` is the requestAnimationFrame timestamp.
/// Returns the stats line.
#[wasm_bindgen]
pub fn frame(now_ms: f64) -> String {
    DEMO.with(|d| {
        let mut d = d.borrow_mut();
        let Some(demo) = d.as_mut() else { return String::new() };
        demo.frame(now_ms)
    })
}

impl Demo {
    fn compile(&mut self) -> Result<(), JsError> {
        let hash = self.graph.topology_hash();
        if self.live == Some(hash) {
            return Ok(());
        }
        let t = web_time::Instant::now();
        if !self.comp.set_graph(&self.graph) {
            return Err(JsError::new("the comp failed to compile (see the console)"));
        }
        self.live = Some(hash);
        if !self.comp.last_compile.cache_hit {
            self.compiles += 1;
            self.compile_ms = t.elapsed().as_secs_f32() * 1e3;
        }
        self.dirty = true;
        Ok(())
    }

    fn frame(&mut self, now_ms: f64) -> String {
        let t = (now_ms - *self.t0.get_or_insert(now_ms)) / 1e3;
        if let Some(NodeKind::Transform(p)) = self.graph.nodes.get_mut(&XFORM) {
            // Sway and breathe: params moving every frame, never a compile.
            p.rotate = self.sway * (t as f32 * 0.5).sin();
            let s = 1.18 + 0.04 * (t as f32 * 0.7).sin();
            p.scale = [s, s];
        }
        self.comp.set_clock((t * 24.0) as f32, 24.0);
        self.comp.write_params(&self.graph);
        self.dirty = false;

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                // Outdated / lost / timed out: reconfigure, skip this frame.
                self.surface.configure(&self.ctx.device, &self.config);
                return String::new();
            }
        };
        let target = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        self.comp.encode_comp(&mut enc, &self.timer);
        // A stage chain built during this encode came up with zero uniforms
        // (CLAUDE.md, Traps): write again before the submit.
        if self.comp.params_stale() {
            self.comp.write_params(&self.graph);
        }
        let (cw, ch) = (self.config.width as f32, self.config.height as f32);
        let s = (cw / W as f32).min(ch / H as f32);
        let (dw, dh) = (W as f32 * s, H as f32 * s);
        self.blit.encode(
            &self.ctx.device,
            &self.ctx.queue,
            &mut enc,
            self.comp.output_view().unwrap(),
            &target,
            None,
            [(cw - dw) * 0.5, (ch - dh) * 0.5, dw, dh],
            false,
            &self.view,
        );
        let stages = [BLUR, GLOW].iter().any(|n| !self.graph.disabled.contains(n));
        self.timer.resolve(&mut enc, false, stages, false);
        self.ctx.queue.submit([enc.finish()]);
        self.timer.after_submit();
        frame.present();
        self.timer.collect();

        let g = self.timer.averages();
        let gpu = if self.timer.enabled() {
            format!("gpu {:.2} ms (stages {:.2}, fused {:.2})", g.prepass_ms + g.fused_ms, g.prepass_ms, g.fused_ms)
        } else {
            "gpu timing unavailable (no timestamp-query)".into()
        };
        let lines = self.comp.current_wgsl().map_or(0, |w| w.lines().count());
        format!(
            "{gpu} · fused pass {lines} lines of generated WGSL · {} compiles, last {:.1} ms",
            self.compiles, self.compile_ms
        )
    }
}

fn demo_graph() -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(READ, NodeKind::read(0, ""));
    g.nodes.insert(
        GRADE,
        NodeKind::Grade(GradeParams { saturation: 1.2, ..Default::default() }),
    );
    g.nodes.insert(XFORM, NodeKind::Transform(TransformParams::default()));
    g.nodes.insert(BLUR, NodeKind::Blur(BlurParams { size: 1.5 }));
    g.nodes.insert(
        GLOW,
        NodeKind::Glow(GlowParams { size: 90.0, threshold: 1.2, intensity: 0.8, ..Default::default() }),
    );
    for (dst, src) in [(GRADE, READ), (XFORM, GRADE), (BLUR, XFORM), (GLOW, BLUR)] {
        g.edges.insert((dst, 0), src);
    }
    g.view = Some(GLOW);
    g
}

fn format_name(f: wgpu::TextureFormat) -> &'static str {
    match f {
        wgpu::TextureFormat::Bgra8Unorm => "bgra8unorm",
        wgpu::TextureFormat::Rgba8Unorm => "rgba8unorm",
        wgpu::TextureFormat::Rgba16Float => "rgba16float",
        _ => "other",
    }
}

/// A scene-linear test plate, built here so the page downloads nothing but
/// the wasm: a dusk sky, a row of colour chips, a fine checker (the blur has
/// something to eat) and a scatter of hot lights well over 1.0 (the glow's
/// threshold is scene-referred, so they are what blooms).
fn plate() -> Vec<[f16; 4]> {
    let (w, h) = (W as f32, H as f32);
    let chips: [[f32; 3]; 8] = [
        [0.40, 0.18, 0.12],
        [0.75, 0.52, 0.40],
        [0.20, 0.28, 0.45],
        [0.18, 0.26, 0.10],
        [0.35, 0.33, 0.62],
        [0.20, 0.58, 0.50],
        [0.85, 0.45, 0.06],
        [0.09, 0.14, 0.52],
    ];
    // Hot lights: deterministic scatter (a tiny LCG), clustered in the sky.
    let mut seed = 0x2545_f491u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    let lights: Vec<([f32; 2], f32, [f32; 3])> = (0..28)
        .map(|_| {
            let x = rnd() * w;
            let y = rnd() * h * 0.55;
            let r = 2.0 + rnd() * 5.0;
            let k = 6.0 + rnd() * 30.0;
            let hue = rnd();
            let c = [k * (0.6 + 0.4 * hue), k * 0.75, k * (1.0 - 0.5 * hue)];
            ([x, y], r, c)
        })
        .collect();

    let mut px = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let v = fy / h;
            // Sky: deep blue overhead into a warm horizon, scene-linear.
            let mut c = [
                0.02 + 0.55 * v * v,
                0.03 + 0.22 * v * v,
                0.10 + 0.06 * v,
            ];
            // Ground: a checker floor below the horizon.
            if v > 0.62 {
                let d = (v - 0.62) / 0.38;
                let u = (fx - w * 0.5) / (fy - h * 0.55);
                let z = 1.0 / d.max(1e-3);
                let check = ((u * 6.0).floor() + (z * 2.0).floor()) as i32 & 1;
                let k = if check == 0 { 0.18 } else { 0.035 };
                let fog = (1.0 - d).powi(3);
                c = [k + fog * 0.3, k * 0.95 + fog * 0.12, k * 0.9 + fog * 0.04];
            }
            // Colour chips in a strip across the horizon.
            let strip_y = h * 0.48;
            if (fy - strip_y).abs() < h * 0.06 {
                let cw = w / 10.0;
                let i = ((fx - cw) / cw).floor();
                let inside = (fx - cw - i * cw) > cw * 0.08 && (fx - cw - i * cw) < cw * 0.92;
                if (0.0..8.0).contains(&i) && inside {
                    c = chips[i as usize];
                }
            }
            // The lights, with a soft core so they read before any glow.
            for (p, r, col) in &lights {
                let d2 = (fx - p[0]).powi(2) + (fy - p[1]).powi(2);
                let fall = (-d2 / (r * r)).exp();
                if fall > 1e-3 {
                    for k in 0..3 {
                        c[k] += col[k] * fall;
                    }
                }
            }
            px.push([
                f16::from_f32(c[0]),
                f16::from_f32(c[1]),
                f16::from_f32(c[2]),
                f16::ONE,
            ]);
        }
    }
    px
}
