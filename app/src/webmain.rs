//! The browser build's glue (`web/`): what start-up and the event loop do
//! differently when wobble runs on WebGPU inside a page. Everything here is
//! wasm32-only; main.rs reaches it from a handful of cfg-gated lines.
//!
//! What a page changes, and where it is answered:
//! - **No blocking.** The adapter and device resolve on the JS event loop,
//!   so `resumed` makes the window and starts [`GpuContext::new_async`]
//!   ([`begin`]); the first event after it lands finishes `build_gfx_with`
//!   ([`ready`]).
//! - **No threads.** rayon falls back to the one thread, where a
//!   `rayon::spawn` job (a decode, say) runs only when that thread yields —
//!   [`pump_rayon`] yields for a slice of every frame.
//! - **No stderr.** `log` goes to the devtools console, panics too.
//! - **The page sizes the canvas**, not the window attributes.
#![cfg(target_arch = "wasm32")]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use engine::GpuContext;
use wasm_bindgen::JsValue;
use winit::event_loop::ActiveEventLoop;
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::{Window, WindowAttributes};

/// Console logging + panics to the console. Called first thing in `main`.
pub fn init() {
    console_error_panic_hook::set_once();
    let _ = log::set_logger(&CONSOLE).map(|()| log::set_max_level(log::LevelFilter::Info));
}

struct Console;
static CONSOLE: Console = Console;

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

/// The desktop attributes, adapted: the page's own `<canvas id="wobble">`,
/// which it sizes (CSS) and pre-sizes (its backing store) BEFORE the app
/// starts. That matters: the viewer holds the picture's screen position
/// when its panel's centre moves, so a canvas that started at the default
/// 300x150 and then grew pushed the picture off-screen. Without that
/// element, one is appended (and the first frames are the wrong size).
pub fn window_attrs(mut a: WindowAttributes) -> WindowAttributes {
    use wasm_bindgen::JsCast;
    a.inner_size = None;
    let canvas = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("wobble"))
        .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok());
    if let Some(c) = &canvas {
        listen_pen(c);
    }
    match canvas {
        Some(c) => a.with_canvas(Some(c)),
        None => a.with_append(true),
    }
}

/// Pen pressure for the brush (`pen.rs`): winit's web backend turns a pen
/// into a mouse and drops `PointerEvent.pressure`, so read it here, in the
/// CAPTURE phase — ahead of winit's own listener on the same element, so
/// the cursor sample it queues carries this event's pressure. The iPad
/// path: Apple Pencil in a browser is `pointerType == "pen"`.
fn listen_pen(canvas: &web_sys::HtmlCanvasElement) {
    use wasm_bindgen::{closure::Closure, JsCast};
    let cb = Closure::<dyn FnMut(web_sys::PointerEvent)>::new(|e: web_sys::PointerEvent| {
        if e.pointer_type() == "pen" {
            let lifted = e.type_() == "pointerup";
            crate::pen::set_pressure(Some(if lifted { 0.0 } else { e.pressure() }));
        } else {
            crate::pen::set_pressure(None);
        }
    });
    for ev in ["pointerdown", "pointermove", "pointerup"] {
        let _ = canvas.add_event_listener_with_callback_and_bool(ev, cb.as_ref().unchecked_ref(), true);
    }
    // Lives as long as the page.
    cb.forget();
}

type Slot = Rc<RefCell<Option<anyhow::Result<GpuContext>>>>;

thread_local! {
    /// The window made in `resumed`, and where its device will land.
    static BOOT: RefCell<Option<(Arc<Window>, Slot)>> = const { RefCell::new(None) };
}

/// `resumed`: make the window and start asking for the device. Idempotent —
/// a second `resumed` while the request is out does nothing.
pub fn begin(event_loop: &ActiveEventLoop, attrs: WindowAttributes) -> anyhow::Result<()> {
    if BOOT.with(|b| b.borrow().is_some()) {
        return Ok(());
    }
    let window = Arc::new(event_loop.create_window(attrs)?);
    let slot: Slot = Rc::default();
    let (w, s) = (window.clone(), slot.clone());
    wasm_bindgen_futures::spawn_local(async move {
        let got = GpuContext::new_async().await;
        *s.borrow_mut() = Some(got);
        // Wake the loop so the next event finishes start-up.
        w.request_redraw();
    });
    BOOT.with(|b| *b.borrow_mut() = Some((window, slot)));
    Ok(())
}

/// The window and its device, once the request has come back AND the canvas
/// has a size (taken: this answers `Some` exactly once). winit reports 0x0
/// until its resize observer fires, and egui lays a 0x0 window out as
/// 10000x10000 points — the viewer remembers that panel centre and holds
/// the picture's screen position against it, which put it thousands of
/// points off-screen. The `Resized` that follows is the event that lands
/// here next.
pub fn ready() -> Option<(Arc<Window>, anyhow::Result<GpuContext>)> {
    BOOT.with(|b| {
        let mut b = b.borrow_mut();
        let (window, slot) = b.as_ref()?;
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return None;
        }
        let got = slot.borrow_mut().take()?;
        let (window, _) = b.take()?;
        Some((window, got))
    })
}

/// Run queued `rayon::spawn` jobs for up to `budget_ms`. On wasm rayon's
/// global pool IS this thread, and a spawned job otherwise waits for a
/// `join` that may never come.
pub fn pump_rayon(budget_ms: f64) {
    let t0 = web_time::Instant::now();
    while t0.elapsed().as_secs_f64() * 1e3 < budget_ms {
        match rayon::yield_now() {
            Some(rayon::Yield::Executed) => {}
            _ => break,
        }
    }
}

/// Say why start-up stopped, on the page as well as in the console: a
/// blank canvas is not an error message.
pub fn fail(msg: &str) {
    log::error!("{msg}");
    let doc = web_sys::window().and_then(|w| w.document());
    if let Some(el) = doc.as_ref().and_then(|d| d.get_element_by_id("boot-error")) {
        el.set_text_content(Some(msg));
        let _ = el.set_attribute("style", "display:grid");
    }
}


/// Redraws the viewer is held framed for after start-up.
const SETTLE_FRAMES: u32 = 30;

/// A page's first frames lay the panels out more than once (egui sizes a
/// side panel from its content), and the viewer holds the picture's SCREEN
/// position against every move of its panel's centre — right for a panel
/// dragged by hand, wrong for a layout still settling: the picture started
/// half out of the viewer. Hold it framed until the layout has settled.
pub fn settle_view(g: &mut crate::Gfx) {
    thread_local! {
        static N: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        static SIZE: std::cell::Cell<(u32, u32)> = const { std::cell::Cell::new((0, 0)) };
    }
    // A resized page (the browser window, a phone turned) is a new layout:
    // frame the picture again rather than carry its old screen position.
    let size = (g.config.width, g.config.height);
    if SIZE.with(|s| s.replace(size)) != size {
        N.with(|n| n.set(0));
    }
    let n = N.with(|n| {
        let v = n.get();
        n.set(v.saturating_add(1));
        v
    });
    if n < SETTLE_FRAMES {
        g.view_pan = egui::Vec2::ZERO;
        g.view_zoom = 1.0;
        g.last_viewer_center = None;
    }
}

// ------------------------------------------------------------------ media
//
// A page has no filesystem and no file dialog, so the page brings the files:
// it puts each one's bytes into the engine's file table (`engine::vfs`) and
// then asks for them to be opened, which lands here as Reads — the same
// `spawn_reads` a pick in the sequence browser takes.

use wasm_bindgen::prelude::wasm_bindgen;

thread_local! {
    static OPENS: RefCell<Vec<Vec<String>>> = const { RefCell::new(Vec::new()) };
    /// Files that came in alone but are numbered like a sequence's frame.
    static LONE_FRAMES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// JS: put one file's bytes in the page's file table, at `path`.
#[wasm_bindgen]
pub fn wobble_put_file(path: String, bytes: Vec<u8>) {
    engine::vfs::insert(path, bytes);
}

thread_local! {
    /// The Read whose "pick files…" opened the picker: what comes back loads
    /// into it instead of making new Reads.
    static PICK_FOR: std::cell::Cell<Option<u64>> = const { std::cell::Cell::new(None) };
}

/// The page's file picker on behalf of one Read node.
pub fn pick_for(node: u64) -> anyhow::Result<()> {
    PICK_FOR.with(|p| p.set(Some(node)));
    pick()
}

/// JS: open these files (already put) as Reads — grouped into sequences by
/// their frame numbers, one Read per sequence. Taken on the next frame.
#[wasm_bindgen]
pub fn wobble_open(paths: Vec<String>) {
    OPENS.with(|o| o.borrow_mut().push(paths));
}

/// Turn queued opens into Reads (and view the first one made); a setup
/// among them (`.wbl`, or a pre-rename `.json`) opens as the comp.
pub fn drain_opens(g: &mut crate::Gfx) {
    let opens = OPENS.with(|o| std::mem::take(&mut *o.borrow_mut()));
    for paths in opens {
        let (setups, media): (Vec<String>, Vec<String>) =
            paths.into_iter().partition(|p| p.ends_with(".wbl") || p.ends_with(".json"));
        for s in &setups {
            g.setup_error = g.load_setup(std::path::Path::new(s)).err().map(|e| format!("{e:#}"));
        }
        if media.is_empty() {
            continue;
        }
        let paths = media;
        let picks = sequences(&paths);
        if picks.is_empty() {
            log::warn!("nothing readable among {} dropped file(s)", paths.len());
            continue;
        }
        let lone: Vec<String> = picks.iter().filter_map(lone_frame).collect();
        if !lone.is_empty() {
            LONE_FRAMES.with(|l| *l.borrow_mut() = lone);
        }
        // A pick for one Read loads into it (its first sequence); a drop, or
        // a pick from the menubar, makes new Reads.
        if let Some(node) = PICK_FOR.with(|p| p.take()).filter(|n| g.egraph.node(*n).is_some()) {
            let p = picks.into_iter().next().unwrap();
            if let Some(crate::NodeKind::Read { path, .. }) = g.egraph.node_mut(node).map(|n| &mut n.kind) {
                *path = p.display.clone();
            }
            g.load_read_files(node, p.display, p.frames, p.missing);
            g.egraph.view = Some(node);
            continue;
        }
        g.spawn_reads(picks);
        if let Some(&id) = g.selected.first() {
            g.egraph.view = Some(id);
            g.egraph.view_out = 0;
            g.egraph.view_active_b = false;
            g.egraph.view_pane = 0;
        }
    }
}

/// A pick of ONE numbered file: most likely one frame of a sequence the
/// page could not see the rest of.
fn lone_frame(p: &crate::browser::Picked) -> Option<String> {
    let [only] = p.frames.as_slice() else { return None };
    if engine::decode::is_movie(only) {
        return None;
    }
    let name = only.file_name()?.to_string_lossy().into_owned();
    name.chars().any(|c| c.is_ascii_digit()).then_some(name)
}

/// Says why a numbered file loaded as one frame, and what to do instead.
pub fn notice_ui(ctx: &egui::Context) {
    let lone = LONE_FRAMES.with(|l| l.borrow().clone());
    if lone.is_empty() {
        return;
    }
    let mut ok = false;
    egui::Window::new("loaded as a single frame")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
        .show(ctx, |ui| {
            ui.set_max_width(380.0);
            let names = match lone.as_slice() {
                [one] => one.clone(),
                [one, rest @ ..] => format!("{one} and {} more", rest.len()),
                [] => String::new(),
            };
            ui.label(format!(
                "{names} loaded as one frame. The web version cannot detect a sequence from a single file: \
                 a browser only lets the page see the files you give it."
            ));
            ui.label("To load a sequence, drop its FOLDER (or select all of its frames at once).");
            ui.weak("This is a limit of the browser, not of wobble: the desktop app finds the whole sequence from any one frame.");
            ok = ui.button("got it").clicked();
        });
    if ok {
        LONE_FRAMES.with(|l| l.borrow_mut().clear());
    }
}

/// Group paths into sequences by the LAST digit run in the file name
/// (`plate.1001.exr`), the rule the browser's scanner uses. Gaps get the
/// path the frame WOULD have and are counted missing: they decode black,
/// as a sparse render does on disk. A name with no digits is a still.
fn sequences(paths: &[String]) -> Vec<crate::browser::Picked> {
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    // (dir, prefix, suffix, pad) -> frame number -> path
    let mut groups: BTreeMap<(String, String, String, usize), BTreeMap<u64, PathBuf>> =
        BTreeMap::new();
    for p in paths {
        let path = PathBuf::from(p);
        if !engine::decode::is_supported(&path) {
            continue;
        }
        let dir = path.parent().map(|d| d.to_string_lossy().into_owned()).unwrap_or_default();
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let end = name.rfind(|c: char| c.is_ascii_digit()).map(|i| i + 1);
        let key_frame = end.and_then(|end| {
            let start = name[..end].rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
            let n = name[start..end].parse::<u64>().ok()?;
            Some(((name[..start].to_string(), name[end..].to_string(), end - start), n))
        });
        let ((prefix, suffix, pad), n) = key_frame.unwrap_or(((name.clone(), String::new(), 0), 0));
        groups.entry((dir, prefix, suffix, pad)).or_default().insert(n, path);
    }
    groups
        .into_iter()
        .map(|((dir, prefix, suffix, pad), frames)| {
            if pad == 0 || frames.len() == 1 {
                let only: Vec<PathBuf> = frames.into_values().collect();
                return crate::browser::Picked {
                    display: only[0].to_string_lossy().into_owned(),
                    frames: only,
                    missing: 0,
                };
            }
            let (first, last) = (*frames.keys().next().unwrap(), *frames.keys().last().unwrap());
            let mut missing = 0;
            let all = (first..=last)
                .map(|n| {
                    frames.get(&n).cloned().unwrap_or_else(|| {
                        missing += 1;
                        PathBuf::from(format!("{dir}/{prefix}{n:0pad$}{suffix}"))
                    })
                })
                .collect();
            crate::browser::Picked {
                display: format!("{dir}/{prefix}{}{suffix}", "#".repeat(pad)),
                frames: all,
                missing,
            }
        })
        .collect()
}

// ------------------------------------------------------------ the browser

/// Where prefs live in a page (`load_prefs` / `save_prefs`).
pub const PREFS_KEY: &str = "wobble.prefs";

fn storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

/// localStorage, which can be absent or throw (a private window, blocked
/// site data): then prefs simply do not persist.
pub fn storage_get(key: &str) -> Option<String> {
    storage()?.get_item(key).ok().flatten()
}

pub fn storage_set(key: &str, value: &str) {
    if let Some(s) = storage() {
        if s.set_item(key, value).is_err() {
            log::warn!("could not save {key} to localStorage");
        }
    }
}

/// A link, in a new tab (`open_url`).
pub fn open_url(url: &str) -> std::io::Result<()> {
    web_sys::window()
        .and_then(|w| w.open_with_url_and_target(url, "_blank").ok().flatten())
        .map(|_| ())
        .ok_or_else(|| std::io::Error::other("the browser blocked the new tab"))
}

/// `load_read` in a page: a path resolved against the page's file table —
/// a `#` pattern (the display a drop gives a sequence), one file, or a
/// directory's frames.
pub fn resolve_read(
    p: &std::path::Path,
    decoder: &dyn engine::decode::ExrDecoder,
) -> anyhow::Result<(engine::player::Sequence, u64)> {
    use engine::player::Sequence;
    let text = p.to_string_lossy();
    if text.contains('#') {
        let dir = p.parent().unwrap_or(std::path::Path::new("/"));
        let listed: Vec<String> = engine::vfs::list(dir)?
            .iter()
            .map(|f| f.to_string_lossy().into_owned())
            .collect();
        let pick = sequences(&listed)
            .into_iter()
            .find(|s| s.display == text)
            .ok_or_else(|| anyhow::anyhow!("{text} is not among this page's files — drop it in"))?;
        return Sequence::from_files(pick.frames, decoder).map(|s| (s, pick.missing));
    }
    if engine::vfs::exists(p) {
        return Sequence::from_files(vec![p.to_path_buf()], decoder).map(|s| (s, 0));
    }
    if !engine::vfs::list(p)?.is_empty() {
        return Sequence::from_dir(p, decoder).map(|s| (s, 0));
    }
    anyhow::bail!("{text} is not among this page's files — drop it in")
}

#[wasm_bindgen]
extern "C" {
    /// The page's download of a document (index.html).
    #[wasm_bindgen(js_namespace = window, js_name = wobbleDownload, catch)]
    fn page_download(name: &str, text: &str) -> Result<(), wasm_bindgen::JsValue>;
    /// The page's file picker; what it picks comes back through `wobble_open`.
    #[wasm_bindgen(js_namespace = window, js_name = wobblePick, catch)]
    fn page_pick() -> Result<(), wasm_bindgen::JsValue>;
}

/// Save: hand the document to the page to download.
pub fn download(name: &str, text: &str) -> anyhow::Result<()> {
    page_download(name, text).map_err(|e| anyhow::anyhow!("download failed: {e:?}"))
}

/// Open: the page's file picker (setups and media alike).
pub fn pick() -> anyhow::Result<()> {
    page_pick().map_err(|e| anyhow::anyhow!("no file picker: {e:?}"))
}

// ----------------------------------------------------------------- movies
//
// The browser decodes movies (WebCodecs); the page's script demuxes each
// dropped MP4/MOV when it arrives and serves decoded frames by index.
// Decoding is asynchronous and a page has no threads to wait on, so a
// frame that is not decoded yet is an error here: the player shows black
// for that tick and asks again, and by then the script has decoded it (it
// works a whole GOP at a time, ahead of the playhead).

#[wasm_bindgen]
extern "C" {
    /// [width, height, frame count] for a movie the page has demuxed.
    #[wasm_bindgen(js_namespace = window, js_name = wobbleMovieInfo)]
    fn movie_info(path: &str) -> Option<Vec<u32>>;
    /// RGBA8 code values of one frame, or nothing yet (it is now asked for).
    #[wasm_bindgen(js_namespace = window, js_name = wobbleMovieFrame)]
    fn movie_frame(path: &str, frame: u32) -> Option<Vec<u8>>;
}

struct WebMovies;

impl WebMovies {
    fn info(path: &std::path::Path) -> anyhow::Result<(u32, u32, u32)> {
        match movie_info(&path.to_string_lossy()).as_deref() {
            Some([w, h, n, ..]) => Ok((*w, *h, *n)),
            _ => anyhow::bail!("{} is not a movie this browser can play", path.display()),
        }
    }
}

impl engine::decode::ExrDecoder for WebMovies {
    fn read_header(&self, path: &std::path::Path) -> anyhow::Result<engine::decode::ImageInfo> {
        let (width, height, _) = Self::info(path)?;
        Ok(engine::decode::ImageInfo { width, height })
    }

    fn frame_count(&self, path: &std::path::Path) -> anyhow::Result<u64> {
        Ok(Self::info(path)?.2 as u64)
    }

    fn decode_into(
        &self,
        path: &std::path::Path,
        info: engine::decode::ImageInfo,
        row_stride: usize,
        dst: &mut [half::f16],
    ) -> anyhow::Result<()> {
        self.decode_frame_into(path, info, row_stride, None, 0, dst)
    }

    fn decode_frame_into(
        &self,
        path: &std::path::Path,
        info: engine::decode::ImageInfo,
        row_stride: usize,
        _layer: Option<&str>,
        frame: u64,
        dst: &mut [half::f16],
    ) -> anyhow::Result<()> {
        let (w, h) = (info.width as usize, info.height as usize);
        anyhow::ensure!(row_stride >= w * 4 && dst.len() >= row_stride * h, "decode dst too small");
        let Some(px) = movie_frame(&path.to_string_lossy(), frame as u32) else {
            anyhow::bail!("{} frame {frame} is still decoding", path.display());
        };
        anyhow::ensure!(px.len() >= w * h * 4, "{} frame {frame}: short picture", path.display());
        // Code values, like every other decoder: the Read's input space
        // (Rec.709 for most movies) turns them into the working space.
        let lut: Vec<half::f16> = (0..256).map(|v| half::f16::from_f32(v as f32 / 255.0)).collect();
        for y in 0..h {
            let src = &px[y * w * 4..(y + 1) * w * 4];
            let row = &mut dst[y * row_stride..y * row_stride + w * 4];
            for (d, s) in row.iter_mut().zip(src) {
                *d = lut[*s as usize];
            }
        }
        Ok(())
    }
}

/// Hand the engine the page's movie decoder (once, at startup).
pub fn install_movies() {
    engine::decode::set_web_movies(Box::new(WebMovies));
}

thread_local! {
    /// Redraws left in which to ask for the shown tick again.
    static MOVIE_READY: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// JS: frames that were asked for and missed have landed.
#[wasm_bindgen]
pub fn wobble_movie_ready() {
    MOVIE_READY.with(|r| r.set(8));
}

/// Show what just landed: the player asks for the tick again. For a few
/// redraws, not one: a black stand-in decoded just before the frame landed
/// can still arrive after the first re-ask and would stay on screen.
pub fn drain_movie_ready(g: &mut crate::Gfx) {
    let left = MOVIE_READY.with(|r| r.get());
    if left > 0 {
        MOVIE_READY.with(|r| r.set(left - 1));
        // The shown tick's movie frames were black stand-ins: forget them
        // so the player decodes them again (the Blender-render precedent).
        let tick = g.timeline_pos;
        if let Some(pb) = g.playback.as_mut() {
            let movies: Vec<usize> =
                pb.player.sequences().iter().enumerate().filter(|(_, s)| s.movie).map(|(i, _)| i).collect();
            for r in movies {
                pb.player.forget(r, tick);
            }
        }
        g.force_display = true;
        g.refresh_display = true;
        g.window.request_redraw();
    }
}
