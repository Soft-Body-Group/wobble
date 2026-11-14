//! The built-in screen recorder (wobble's, ported): a clip of Scrub itself, taken from the
//! frames it already draws, trimmed, then handed to the user to share.
//!
//! Capture is GPU frames, not an OS screen grab: the surface texture is
//! copied into a readback buffer right before the submit that draws it, so
//! there is no screen-recording permission, nothing else on the desktop is
//! ever in the picture, and it works the same on every platform. Only while
//! recording does the surface carry `COPY_SRC` (the app flips it in and out,
//! because on Metal it costs the swapchain its framebuffer-only fast path).
//!
//! Frames go to a worker thread that crops, box-downscales to fit
//! [`MAX_W`]x[`MAX_H`], LZ4s them and appends them to a temp file â€” so a
//! take lives on disk, not in RAM, and the trim is exact (every captured
//! frame is still there). The cursor and the "made with wobble" mark are
//! laid on at SAVE time, so both can be switched after the fact.
//!
//! Saving encodes with the OPERATING SYSTEM's H.264 encoder (AVFoundation
//! on macOS, Media Foundation on Windows; `encode_mac` / `encode_win`), so
//! no GPL encoder ships and the public build needs no FFmpeg for it.
//! Elsewhere it pipes to an `ffmpeg` the user has installed, or writes the
//! GIF alone. Sharing uploads nothing on its own: the share panel opens the
//! OS share sheet, reveals or copies the file, or opens a site's compose
//! page â€” the user attaches the file there. The
//! one upload is the wobble blog's, from its own dialog (`blogpost.rs`).

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[cfg(target_os = "macos")]
mod encode_mac;
#[cfg(target_os = "windows")]
mod encode_win;
mod overlay;
pub mod share;
pub mod shot;

/// Capture rate. A redraw that comes sooner than this after the last
/// captured one is skipped; a slow one is simply a longer frame (every
/// frame carries its own time, and the encoders take variable timing).
pub const FPS: f64 = 30.0;
/// A take is downscaled (by a whole factor, box filtered) to fit this.
pub const MAX_W: u32 = 1920;
pub const MAX_H: u32 = 1200;
/// A take stops itself here, so a forgotten recording cannot fill a disk.
pub const MAX_SECS: f64 = 300.0;
/// GIFs are big; they get a smaller picture and half the rate.
const GIF_MAX_W: u32 = 720;
const GIF_FPS: f64 = 15.0;
/// Readback buffers in flight. A redraw that finds none free drops its
/// frame (counted, shown) rather than stalling the GPU on a map.
const IN_FLIGHT: usize = 3;
const THUMB_H: u32 = 72;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Area {
    /// The whole Scrub window.
    #[default]
    Window,
    /// The viewer pane alone (its rect when the recording starts).
    Viewer,
}

/// What the user picked, kept in prefs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Opts {
    pub area: Area,
    pub cursor: bool,
    pub watermark: bool,
    pub gif: bool,
    /// Where clips are saved; empty = the platform's videos folder.
    pub clip_dir: String,
    /// Where screenshots are saved; empty = the platform's pictures folder.
    pub shot_dir: String,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            area: Area::Window,
            cursor: true,
            watermark: true,
            gif: false,
            clip_dir: String::new(),
            shot_dir: String::new(),
        }
    }
}

/// The folders the user picked (clips, screenshots), read by
/// `share::save_dir` / `shot::save_dir`. Empty = the default.
static DIRS: Mutex<(String, String)> = Mutex::new((String::new(), String::new()));

fn set_dirs(o: &Opts) {
    if let Ok(mut d) = DIRS.lock() {
        *d = (o.clip_dir.clone(), o.shot_dir.clone());
    }
}

/// The picked folder for clips (`shot` false) or screenshots, if any.
/// `EDIFY_RECORD_DIR` wins over both.
pub(crate) fn picked_dir(shot: bool) -> Option<PathBuf> {
    if let Ok(d) = media::sys::var("RECORD_DIR") {
        return Some(PathBuf::from(d));
    }
    let d = DIRS.lock().ok()?;
    let s = if shot { &d.1 } else { &d.0 };
    let s = s.trim();
    (!s.is_empty()).then(|| imfram::filebrowser::expand_home(s))
}

/// One captured frame in the take file.
#[derive(Clone, Copy, Debug)]
struct FrameRec {
    offset: u64,
    len: u32,
    /// Seconds since the recording started.
    t: f64,
    /// Pointer in OUTPUT pixels, if it was over the window.
    cursor: Option<[f32; 2]>,
}

/// The take as the worker builds it; the review reads it.
#[derive(Default)]
struct TakeShared {
    frames: Vec<FrameRec>,
    /// One small RGBA picture per frame, for the trim strip.
    thumbs: Vec<(u32, u32, Vec<u8>)>,
    error: Option<String>,
}

/// A take on disk.
struct Take {
    dir: PathBuf,
    file: PathBuf,
    w: u32,
    h: u32,
    /// Output pixels per egui point (for drawing a cursor the right size).
    px_per_pt: f32,
    shared: Arc<Mutex<TakeShared>>,
}

impl Drop for Take {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Take {
    fn read_frame(&self, i: usize) -> Option<Vec<u8>> {
        let rec = *self.shared.lock().ok()?.frames.get(i)?;
        let mut f = std::fs::File::open(&self.file).ok()?;
        f.seek(SeekFrom::Start(rec.offset)).ok()?;
        let mut buf = vec![0u8; rec.len as usize];
        f.read_exact(&mut buf).ok()?;
        lz4_flex::decompress_size_prepended(&buf).ok()
    }
}

/// Raw pixels off the GPU, on their way to the worker.
struct RawFrame {
    data: Vec<u8>,
    row_bytes: usize,
    bgra: bool,
    t: f64,
    /// Pointer in CROP pixels.
    cursor: Option<[f32; 2]>,
}

/// Where a capture crops the surface and how it scales down.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    /// Crop rect in surface pixels.
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    /// Whole downscale factor.
    pub k: u32,
    /// Output size: crop / k, rounded down to even (H.264 wants even).
    pub out_w: u32,
    pub out_h: u32,
}

impl Geometry {
    /// The crop rect clamped into a `sw`x`sh` surface, and the factor that
    /// fits it inside [`MAX_W`]x[`MAX_H`]. `None` when nothing is left.
    pub fn new(rect: [f32; 4], sw: u32, sh: u32) -> Option<Self> {
        let x0 = rect[0].max(0.0).round() as u32;
        let y0 = rect[1].max(0.0).round() as u32;
        let x1 = (rect[0] + rect[2]).round().clamp(0.0, sw as f32) as u32;
        let y1 = (rect[1] + rect[3]).round().clamp(0.0, sh as f32) as u32;
        if x1 <= x0 + 16 || y1 <= y0 + 16 {
            return None;
        }
        let (w, h) = (x1 - x0, y1 - y0);
        let k = w.div_ceil(MAX_W).max(h.div_ceil(MAX_H)).max(1);
        let out_w = (w / k) & !1;
        let out_h = (h / k) & !1;
        Some(Self { x: x0, y: y0, w: out_w * k, h: out_h * k, k, out_w, out_h })
    }
}

struct Slot {
    buf: wgpu::Buffer,
    /// 0 free, 1 copy recorded, 2 mapping, 3 mapped, 4 failed.
    state: Arc<AtomicU32>,
    t: f64,
    cursor: Option<[f32; 2]>,
    bgra: bool,
}

struct Recording {
    geo: Geometry,
    row_bytes: u32,
    slots: Vec<Slot>,
    started: Instant,
    next_t: f64,
    dropped: u32,
    /// The surface is no longer big enough for the crop (a resize).
    paused: bool,
    tx: mpsc::Sender<Option<RawFrame>>,
    worker: Option<std::thread::JoinHandle<()>>,
    take: Take,
}

/// The save job's progress and answer.
struct SaveJob {
    done: Arc<AtomicUsize>,
    total: usize,
    cancel: Arc<AtomicBool>,
    result: Arc<Mutex<Option<Result<Saved, String>>>>,
}

#[derive(Clone, Debug)]
pub struct Saved {
    pub mp4: Option<PathBuf>,
    pub gif: Option<PathBuf>,
    /// Said under the result: why there is no MP4, say.
    pub note: Option<String>,
}

struct Review {
    take: Take,
    /// Trim, as frame indices (inclusive).
    a: usize,
    b: usize,
    /// The frame in the preview.
    at: usize,
    shown: Option<(usize, bool, bool)>,
    preview: Option<egui::TextureHandle>,
    strip: Vec<egui::TextureHandle>,
    job: Option<SaveJob>,
    saved: Option<Saved>,
    error: Option<String>,
    drag: Option<Handle>,
}

#[derive(Clone, Copy, PartialEq)]
enum Handle {
    A,
    B,
    Scrub,
}

enum State {
    Idle,
    Recording(Box<Recording>),
    Review(Box<Review>),
}

pub struct Recorder {
    state: State,
    pub opts: Opts,
    /// The options changed: save prefs.
    pub opts_dirty: bool,
    /// Last start failure, shown by the button.
    pub error: Option<String>,
    /// Remembered so a share sheet has somewhere to anchor.
    share_anchor: Option<egui::Rect>,
}

impl Recorder {
    pub fn new(opts: Opts) -> Self {
        set_dirs(&opts);
        Self { state: State::Idle, opts, opts_dirty: false, error: None, share_anchor: None }
    }

    pub fn recording(&self) -> bool {
        matches!(self.state, State::Recording(_))
    }

    /// Seconds recorded so far.
    pub fn elapsed(&self) -> Option<f64> {
        match &self.state {
            State::Recording(r) => Some(r.started.elapsed().as_secs_f64()),
            _ => None,
        }
    }

    /// Begin a take of `rect` (surface pixels) on a `sw`x`sh` surface.
    /// `px_per_pt` is the surface's pixels per egui point.
    pub fn start(&mut self, device: &wgpu::Device, rect: [f32; 4], sw: u32, sh: u32, px_per_pt: f32) {
        self.error = None;
        if !matches!(self.state, State::Idle | State::Review(_)) {
            return;
        }
        let Some(geo) = Geometry::new(rect, sw, sh) else {
            self.error = Some("nothing to record: the area is empty".into());
            return;
        };
        let dir = std::env::temp_dir().join(format!(
            "scrub-rec-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        ));
        if let Err(e) = std::fs::create_dir_all(&dir) {
            self.error = Some(format!("cannot make {}: {e}", dir.display()));
            return;
        }
        let file = dir.join("frames.lz4");
        let out = match std::fs::File::create(&file) {
            Ok(f) => f,
            Err(e) => {
                self.error = Some(format!("cannot write {}: {e}", file.display()));
                return;
            }
        };
        let row_bytes = (geo.w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let slots = (0..IN_FLIGHT)
            .map(|i| Slot {
                buf: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("rec-readback-{i}")),
                    size: row_bytes as u64 * geo.h as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU32::new(0)),
                t: 0.0,
                cursor: None,
                bgra: true,
            })
            .collect();
        let shared = Arc::new(Mutex::new(TakeShared::default()));
        let (tx, rx) = mpsc::channel::<Option<RawFrame>>();
        let worker = {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("rec-worker".into())
                .spawn(move || worker(rx, out, geo, shared))
                .ok()
        };
        let take = Take { dir, file, w: geo.out_w, h: geo.out_h, px_per_pt: px_per_pt / geo.k as f32, shared };
        self.state = State::Recording(Box::new(Recording {
            geo,
            row_bytes,
            slots,
            started: Instant::now(),
            next_t: 0.0,
            dropped: 0,
            paused: false,
            tx,
            worker,
            take,
        }));
    }

    /// End the take and open the review.
    pub fn stop(&mut self, device: &wgpu::Device) {
        let State::Recording(mut rec) = std::mem::replace(&mut self.state, State::Idle) else {
            return;
        };
        // Whatever is still on its way back from the GPU is waited for, so
        // the last second is not lost.
        let deadline = Instant::now() + std::time::Duration::from_secs(2);
        while rec.slots.iter().any(|s| matches!(s.state.load(Ordering::Acquire), 1 | 2)) && Instant::now() < deadline {
            map_recorded(&mut rec.slots);
            let _ = device.poll(wgpu::PollType::Poll);
            collect(&mut rec);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        collect(&mut rec);
        let _ = rec.tx.send(None);
        if let Some(w) = rec.worker.take() {
            let _ = w.join();
        }
        let n = rec.take.shared.lock().map(|s| s.frames.len()).unwrap_or(0);
        let err = rec.take.shared.lock().ok().and_then(|s| s.error.clone());
        if n == 0 {
            self.error = Some(err.unwrap_or_else(|| "no frames were captured".into()));
            return;
        }
        let rec = *rec;
        self.state = State::Review(Box::new(Review {
            take: rec.take,
            a: 0,
            b: n - 1,
            at: 0,
            shown: None,
            preview: None,
            strip: Vec::new(),
            job: None,
            saved: None,
            error: err,
            drag: None,
        }));
    }

    /// Record this frame's copy, if one is due. Call with the encoder that
    /// draws `texture`, after everything that draws into it.
    pub fn capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
        cursor_px: Option<[f32; 2]>,
    ) {
        let State::Recording(rec) = &mut self.state else { return };
        let t = rec.started.elapsed().as_secs_f64();
        if t < rec.next_t {
            return;
        }
        let bgra = match format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
            _ => {
                rec.paused = true;
                return;
            }
        };
        let g = rec.geo;
        let size = texture.size();
        rec.paused = g.x + g.w > size.width || g.y + g.h > size.height;
        if rec.paused {
            return;
        }
        // Keep to the clock, but never try to catch up a backlog.
        rec.next_t = if t - rec.next_t > 1.0 / FPS { t + 1.0 / FPS } else { rec.next_t + 1.0 / FPS };
        let Some(slot) = rec.slots.iter_mut().find(|s| s.state.load(Ordering::Acquire) == 0) else {
            rec.dropped += 1;
            return;
        };
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: g.x, y: g.y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(rec.row_bytes),
                    rows_per_image: Some(g.h),
                },
            },
            wgpu::Extent3d { width: g.w, height: g.h, depth_or_array_layers: 1 },
        );
        slot.t = t;
        slot.bgra = bgra;
        slot.cursor = cursor_px.map(|[x, y]| [x - g.x as f32, y - g.y as f32]);
        slot.state.store(1, Ordering::Release);
    }

    /// After the submit that carried this frame's copy.
    pub fn after_submit(&mut self) {
        if let State::Recording(rec) = &mut self.state {
            map_recorded(&mut rec.slots);
        }
    }

    /// Once a frame: hand landed readbacks to the worker; stop at the cap.
    /// Returns true when the take stopped itself.
    pub fn poll(&mut self, device: &wgpu::Device) -> bool {
        let State::Recording(rec) = &mut self.state else { return false };
        let _ = device.poll(wgpu::PollType::Poll);
        collect(rec);
        rec.started.elapsed().as_secs_f64() >= MAX_SECS
    }

    /// Frames in the take under review.
    pub fn review_frames(&self) -> Option<usize> {
        match &self.state {
            State::Review(rv) => rv.take.shared.lock().ok().map(|s| s.frames.len()),
            _ => None,
        }
    }

    /// Save the whole take under review, as the save button does.
    pub fn save_all(&mut self) {
        if let State::Review(rv) = &mut self.state {
            if rv.job.is_none() {
                rv.saved = None;
                rv.job = Some(start_save(&rv.take, rv.a, rv.b, &self.opts, share::save_dir()));
            }
        }
    }

    /// A save is running.
    pub fn saving(&self) -> bool {
        matches!(&self.state, State::Review(rv) if rv.job.is_some())
    }

    /// The last save's answer, once there is one.
    pub fn save_result(&self) -> Option<Result<Saved, String>> {
        match &self.state {
            State::Review(rv) if rv.job.is_none() => match (&rv.saved, &rv.error) {
                (Some(s), _) => Some(Ok(s.clone())),
                (None, Some(e)) => Some(Err(e.clone())),
                _ => None,
            },
            _ => None,
        }
    }

    /// The menubar control: a dot to start, a square and the time to stop.
    /// Returns true when clicked.
    pub fn bar_button(&self, ui: &mut egui::Ui, key: &str) -> bool {
        let rec = self.elapsed();
        let text = match rec {
            Some(s) => format!("{}:{:02}", (s as u64) / 60, (s as u64) % 60),
            None => "rec".into(),
        };
        let galley = ui.painter().layout_no_wrap(text, egui::FontId::proportional(12.0), ui.visuals().text_color());
        let size = egui::vec2(galley.size().x + 22.0, ui.spacing().interact_size.y);
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
        let vis = ui.style().interact(&resp);
        ui.painter().rect_filled(rect, 0.0, vis.weak_bg_fill);
        let red = egui::Color32::from_rgb(230, 80, 70);
        let c = egui::pos2(rect.min.x + 10.0, rect.center().y);
        match rec {
            Some(s) => {
                // Breathes once a second so a running take is hard to miss.
                let a = 0.55 + 0.45 * (s * std::f64::consts::TAU).cos().abs() as f32;
                ui.painter().rect_filled(egui::Rect::from_center_size(c, egui::vec2(8.0, 8.0)), 1.0, red.gamma_multiply(a));
                ui.ctx().request_repaint();
            }
            None => {
                ui.painter().circle_filled(c, 4.5, red);
            }
        }
        ui.painter().galley(egui::pos2(rect.min.x + 18.0, rect.center().y - galley.size().y / 2.0), galley, vis.text_color());
        let tip = match rec {
            Some(_) => format!("stop recording Â· {key}"),
            None => format!(
                "record a clip of Scrub Â· {key}\nright-click for options (window or viewer, cursor, watermark, save folders)"
            ),
        };
        let resp = resp.on_hover_text(tip);
        let mut opts = self.opts.clone();
        resp.context_menu(|ui| options_ui(ui, &mut opts, true));
        if opts != self.opts {
            // A context menu borrows immutably; the change rides a temp.
            ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new("rec-opts"), opts));
        }
        resp.clicked()
    }

    /// The review / share window, while there is a take. Call once a frame
    /// after `bar_button`. `view` is the native view, for the share sheet.
    pub fn ui(&mut self, ctx: &egui::Context, window: Option<&winit::window::Window>) {
        if let Some(o) = ctx.data_mut(|d| d.remove_temp::<Opts>(egui::Id::new("rec-opts"))) {
            self.opts = o;
            set_dirs(&self.opts);
            self.opts_dirty = true;
        }
        if !matches!(self.state, State::Review(_)) {
            return;
        }
        let mut open = true;
        let mut discard = false;
        let opts_before = self.opts.clone();
        egui::Window::new("recording")
            .id(egui::Id::new("rec-review"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_pos(ctx.content_rect().center() - egui::vec2(260.0, 220.0))
            .show(ctx, |ui| {
                let State::Review(rv) = &mut self.state else { return };
                discard = review_ui(ui, rv, &mut self.opts, &mut self.share_anchor, window);
            });
        if self.opts != opts_before {
            set_dirs(&self.opts);
            self.opts_dirty = true;
        }
        if !open || discard {
            if let State::Review(rv) = &self.state {
                if let Some(j) = &rv.job {
                    j.cancel.store(true, Ordering::Release);
                }
            }
            self.state = State::Idle;
        }
    }
}

/// Map every slot whose copy was just submitted.
fn map_recorded(slots: &mut [Slot]) {
    for s in slots {
        if s.state.load(Ordering::Acquire) == 1 {
            s.state.store(2, Ordering::Release);
            let st = s.state.clone();
            s.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                st.store(if r.is_ok() { 3 } else { 4 }, Ordering::Release);
            });
        }
    }
}

/// Copy landed readbacks out, free the slots, feed the worker in time order.
fn collect(rec: &mut Recording) {
    let mut landed: Vec<RawFrame> = Vec::new();
    for s in &mut rec.slots {
        match s.state.load(Ordering::Acquire) {
            3 => {
                let Ok(view) = s.buf.slice(..).get_mapped_range() else { continue };
                let data = view.to_vec();
                drop(view);
                s.buf.unmap();
                landed.push(RawFrame { data, row_bytes: rec.row_bytes as usize, bgra: s.bgra, t: s.t, cursor: s.cursor });
                s.state.store(0, Ordering::Release);
            }
            4 => {
                s.buf.unmap();
                rec.dropped += 1;
                s.state.store(0, Ordering::Release);
            }
            _ => {}
        }
    }
    landed.sort_by(|a, b| a.t.total_cmp(&b.t));
    for f in landed {
        let _ = rec.tx.send(Some(f));
    }
}

/// Crop is done by the copy; this downscales by `k` (box), converts to
/// RGBA and drops the row padding.
fn shrink(raw: &RawFrame, geo: &Geometry) -> Vec<u8> {
    use rayon::prelude::*;
    let (ow, oh, k) = (geo.out_w as usize, geo.out_h as usize, geo.k as usize);
    let mut out = vec![0u8; ow * oh * 4];
    let (r, b) = if raw.bgra { (2, 0) } else { (0, 2) };
    out.par_chunks_mut(ow * 4).enumerate().for_each(|(y, row)| {
        for x in 0..ow {
            let mut acc = [0u32; 3];
            for dy in 0..k {
                let src = &raw.data[(y * k + dy) * raw.row_bytes + x * k * 4..];
                for dx in 0..k {
                    let p = &src[dx * 4..dx * 4 + 4];
                    acc[0] += p[r] as u32;
                    acc[1] += p[1] as u32;
                    acc[2] += p[b] as u32;
                }
            }
            let n = (k * k) as u32;
            let o = &mut row[x * 4..x * 4 + 4];
            o[0] = ((acc[0] + n / 2) / n) as u8;
            o[1] = ((acc[1] + n / 2) / n) as u8;
            o[2] = ((acc[2] + n / 2) / n) as u8;
            o[3] = 255;
        }
    });
    out
}

/// Nearest-ish thumbnail (area average over a grid) for the trim strip.
fn thumb(rgba: &[u8], w: u32, h: u32, th: u32) -> (u32, u32, Vec<u8>) {
    let tw = ((w as f32 * th as f32 / h as f32).round() as u32).max(1);
    let mut out = vec![0u8; (tw * th * 4) as usize];
    for y in 0..th {
        for x in 0..tw {
            let sx = (x * w / tw).min(w - 1);
            let sy = (y * h / th).min(h - 1);
            let i = ((sy * w + sx) * 4) as usize;
            let o = ((y * tw + x) * 4) as usize;
            out[o..o + 4].copy_from_slice(&rgba[i..i + 4]);
        }
    }
    (tw, th, out)
}

fn worker(rx: mpsc::Receiver<Option<RawFrame>>, out: std::fs::File, geo: Geometry, shared: Arc<Mutex<TakeShared>>) {
    let mut out = std::io::BufWriter::with_capacity(1 << 22, out);
    let mut offset = 0u64;
    while let Ok(Some(raw)) = rx.recv() {
        let rgba = shrink(&raw, &geo);
        let packed = lz4_flex::compress_prepend_size(&rgba);
        if let Err(e) = out.write_all(&packed) {
            if let Ok(mut s) = shared.lock() {
                s.error.get_or_insert(format!("writing the take: {e}"));
            }
            continue;
        }
        let th = thumb(&rgba, geo.out_w, geo.out_h, THUMB_H);
        let k = geo.k as f32;
        let rec = FrameRec {
            offset,
            len: packed.len() as u32,
            t: raw.t,
            cursor: raw.cursor.map(|[x, y]| [x / k, y / k]),
        };
        offset += packed.len() as u64;
        if let Ok(mut s) = shared.lock() {
            s.frames.push(rec);
            s.thumbs.push(th);
        }
    }
    let _ = out.flush();
}

/// The options, in the button's context menu and in the review.
fn options_ui(ui: &mut egui::Ui, o: &mut Opts, area: bool) {
    if area {
        ui.horizontal(|ui| {
            ui.label("record");
            ui.selectable_value(&mut o.area, Area::Window, "window");
            ui.selectable_value(&mut o.area, Area::Viewer, "viewer");
        });
    }
    ui.checkbox(&mut o.cursor, "show the pointer");
    ui.checkbox(&mut o.watermark, "\"made with Scrub\" mark");
    ui.checkbox(&mut o.gif, "also save a GIF");
    if area {
        ui.separator();
        dir_row(ui, "clips to", &mut o.clip_dir, share::default_dir());
        dir_row(ui, "screenshots to", &mut o.shot_dir, shot::default_dir());
        if media::sys::var_os("RECORD_DIR").is_some() {
            ui.weak("SCRUB_RECORD_DIR (or EDIFY_RECORD_DIR) is set and wins over both");
        }
    }
}

/// A save folder: the path as typed (empty = `default`, shown greyed),
/// a folder picker, and a reset.
fn dir_row(ui: &mut egui::Ui, label: &str, dir: &mut String, default: PathBuf) {
    ui.label(label);
    ui.horizontal(|ui| {
        let hint = default.display().to_string();
        ui.add(egui::TextEdit::singleline(dir).hint_text(hint).desired_width(220.0));
        if ui.button("choose…").clicked() {
            let start = if dir.trim().is_empty() { default.clone() } else { imfram::filebrowser::expand_home(dir.trim()) };
            if let Some(p) = imfram::filedialog::FileDialog::new().set_directory(&start).pick_folder() {
                *dir = p.display().to_string();
            }
        }
        if ui.add_enabled(!dir.is_empty(), egui::Button::new("default")).clicked() {
            dir.clear();
        }
    });
}

fn fmt_t(t: f64) -> String {
    format!("{}:{:04.1}", (t as u64) / 60, t % 60.0)
}

/// The review window's body. Returns true to discard the take.
fn review_ui(
    ui: &mut egui::Ui,
    rv: &mut Review,
    opts: &mut Opts,
    anchor: &mut Option<egui::Rect>,
    window: Option<&winit::window::Window>,
) -> bool {
    let (n, times) = {
        let s = rv.take.shared.lock().unwrap();
        (s.frames.len(), (s.frames.first().map(|f| f.t).unwrap_or(0.0), s.frames.last().map(|f| f.t).unwrap_or(0.0)))
    };
    let t_of = |i: usize| rv.take.shared.lock().ok().and_then(|s| s.frames.get(i).map(|f| f.t)).unwrap_or(0.0);
    let busy = rv.job.is_some();

    // Preview: the full frame under the scrub, with the overlays as they
    // will be saved.
    let want = (rv.at, opts.cursor, opts.watermark);
    if rv.shown != Some(want) {
        if let Some(mut px) = rv.take.read_frame(rv.at) {
            let cursor = rv.take.shared.lock().ok().and_then(|s| s.frames.get(rv.at).and_then(|f| f.cursor));
            overlay::apply(&mut px, rv.take.w, rv.take.h, cursor.filter(|_| opts.cursor), rv.take.px_per_pt, opts.watermark);
            let img = egui::ColorImage::from_rgba_unmultiplied([rv.take.w as usize, rv.take.h as usize], &px);
            match &mut rv.preview {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => rv.preview = Some(ui.ctx().load_texture("rec-preview", img, egui::TextureOptions::LINEAR)),
            }
            rv.shown = Some(want);
        }
    }
    let pw = 520.0;
    let ph = pw * rv.take.h as f32 / rv.take.w as f32;
    if let Some(t) = &rv.preview {
        ui.add(egui::Image::new((t.id(), egui::vec2(pw, ph))));
    }

    // The trim strip: thumbnails, the kept span lit, a handle each end, a
    // scrub line. Drag a handle to trim, anywhere else to scrub.
    if rv.strip.is_empty() {
        let s = rv.take.shared.lock().unwrap();
        let tiles = 10usize.min(s.thumbs.len()).max(1);
        for j in 0..tiles {
            let i = j * s.thumbs.len() / tiles;
            let (w, h, px) = &s.thumbs[i];
            let img = egui::ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], px);
            rv.strip.push(ui.ctx().load_texture(format!("rec-strip-{j}"), img, egui::TextureOptions::LINEAR));
        }
    }
    let sh = 40.0;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(pw, sh), egui::Sense::click_and_drag());
    let p = ui.painter_at(rect);
    let tw = pw / rv.strip.len() as f32;
    for (j, t) in rv.strip.iter().enumerate() {
        let r = egui::Rect::from_min_size(rect.min + egui::vec2(j as f32 * tw, 0.0), egui::vec2(tw, sh));
        p.image(t.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), egui::Color32::WHITE);
    }
    let last = (n - 1).max(1) as f32;
    let x_of = |i: usize| rect.min.x + rect.width() * i as f32 / last;
    let i_of = |x: f32| (((x - rect.min.x) / rect.width()).clamp(0.0, 1.0) * last).round() as usize;
    let shade = egui::Color32::from_black_alpha(170);
    p.rect_filled(egui::Rect::from_min_max(rect.min, egui::pos2(x_of(rv.a), rect.max.y)), 0.0, shade);
    p.rect_filled(egui::Rect::from_min_max(egui::pos2(x_of(rv.b), rect.min.y), rect.max), 0.0, shade);
    let accent = ui.visuals().selection.bg_fill;
    for i in [rv.a, rv.b] {
        let x = x_of(i);
        p.rect_filled(egui::Rect::from_min_max(egui::pos2(x - 2.0, rect.min.y), egui::pos2(x + 2.0, rect.max.y)), 1.0, accent);
    }
    p.line_segment(
        [egui::pos2(x_of(rv.at), rect.min.y), egui::pos2(x_of(rv.at), rect.max.y)],
        egui::Stroke::new(1.5, egui::Color32::WHITE),
    );
    if !busy {
        if resp.drag_started() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let near = |i: usize| (x_of(i) - pos.x).abs() < 8.0;
                rv.drag = Some(if near(rv.a) { Handle::A } else if near(rv.b) { Handle::B } else { Handle::Scrub });
            }
        }
        if let (Some(h), Some(pos)) = (rv.drag, resp.interact_pointer_pos()) {
            if resp.dragged() || resp.clicked() {
                let i = i_of(pos.x);
                match h {
                    Handle::A => rv.a = i.min(rv.b),
                    Handle::B => rv.b = i.max(rv.a),
                    Handle::Scrub => {}
                }
                rv.at = i.clamp(rv.a, rv.b);
            }
        } else if resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                rv.at = i_of(pos.x).clamp(rv.a, rv.b);
            }
        }
        if resp.drag_stopped() {
            rv.drag = None;
        }
    }
    let (ta, tb) = (t_of(rv.a), t_of(rv.b));
    ui.horizontal(|ui| {
        ui.weak(format!(
            "keeping {} of {}, from {} to {}",
            fmt_t(tb - ta + 1.0 / FPS),
            fmt_t(times.1 - times.0 + 1.0 / FPS),
            fmt_t(ta - times.0),
            fmt_t(tb - times.0)
        ));
    });
    ui.weak("drag the ends of the strip to trim, anywhere else to look");
    ui.add_space(4.0);
    ui.add_enabled_ui(!busy, |ui| options_ui(ui, opts, false));

    let mut discard = false;
    ui.add_space(6.0);
    if let Some(job) = &rv.job {
        let done = job.done.load(Ordering::Acquire);
        ui.add(egui::ProgressBar::new(done as f32 / job.total.max(1) as f32).text(format!("encoding {done}/{}", job.total)));
        ui.ctx().request_repaint();
        let result = job.result.lock().ok().and_then(|mut r| r.take());
        if let Some(r) = result {
            rv.job = None;
            match r {
                Ok(s) => {
                    rv.saved = Some(s);
                    rv.error = None;
                }
                Err(e) => {
                    rv.error = Some(e);
                }
            }
        }
    } else {
        ui.horizontal(|ui| {
            let label = if rv.saved.is_some() { "save again" } else { "save" };
            if ui.button(label).on_hover_text(format!("into {}", share::save_dir().display())).clicked() {
                rv.saved = None;
                rv.job = Some(start_save(&rv.take, rv.a, rv.b, opts, share::save_dir()));
            }
            if ui.button("discard").clicked() {
                discard = true;
            }
        });
    }
    if let Some(e) = &rv.error {
        ui.colored_label(egui::Color32::from_rgb(230, 80, 70), e);
    }
    if let Some(saved) = &rv.saved {
        ui.separator();
        let _ = (anchor, window);
        if let Some(n) = &saved.note {
            ui.weak(n);
        }
        for p in saved.mp4.iter().chain(saved.gif.iter()) {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if ui.link(format!("saved {name}  (show)")).on_hover_text(p.display().to_string()).clicked() {
                share::reveal(p);
            }
        }
    }
    discard
}

fn start_save(take: &Take, a: usize, b: usize, opts: &Opts, dir: PathBuf) -> SaveJob {
    let frames: Vec<FrameRec> = take.shared.lock().map(|s| s.frames[a..=b].to_vec()).unwrap_or_default();
    let done = Arc::new(AtomicUsize::new(0));
    let cancel = Arc::new(AtomicBool::new(false));
    let result = Arc::new(Mutex::new(None));
    let job = SaveJob { done: done.clone(), total: frames.len(), cancel: cancel.clone(), result: result.clone() };
    let (file, w, h, ppp) = (take.file.clone(), take.w, take.h, take.px_per_pt);
    let opts = opts.clone();
    let spawned = std::thread::Builder::new().name("rec-save".into()).spawn(move || {
        let r = save(&dir, &file, &frames, w, h, ppp, &opts, &done, &cancel);
        if let Ok(mut slot) = result.lock() {
            *slot = Some(r);
        }
    });
    if let Err(e) = spawned {
        if let Ok(mut slot) = job.result.lock() {
            *slot = Some(Err(format!("could not start the encoder: {e}")));
        }
    }
    job
}

/// Something that takes RGBA frames at given times and writes a movie.
pub trait VideoSink {
    fn push(&mut self, rgba: &[u8], t: f64) -> Result<(), String>;
    fn finish(self: Box<Self>, end: f64) -> Result<(), String>;
}

/// The platform's own H.264 encoder into an .mp4, or why there is none.
pub(crate) fn open_mp4(path: &Path, w: u32, h: u32) -> Result<Box<dyn VideoSink>, String> {
    // ~0.15 bits a pixel at 30fps: plenty for UI, which is mostly flat.
    let bitrate = ((w as f64 * h as f64 * FPS * 0.15) as u64).clamp(2_000_000, 24_000_000);
    #[cfg(target_os = "macos")]
    {
        encode_mac::open(path, w, h, bitrate)
    }
    #[cfg(target_os = "windows")]
    {
        encode_win::open(path, w, h, bitrate)
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        CliSink::open(path, w, h, bitrate)
    }
}

/// Linux and friends: an `ffmpeg` the user installed, fed raw frames at a
/// constant rate. Wobble ships no encoder here.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
struct CliSink {
    child: std::process::Child,
    next: f64,
    last: Vec<u8>,
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
impl CliSink {
    fn open(path: &Path, w: u32, h: u32, bitrate: u64) -> Result<Box<dyn VideoSink>, String> {
        let child = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s"])
            .arg(format!("{w}x{h}"))
            .args(["-r", "30", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-b:v"])
            .arg(bitrate.to_string())
            .arg(path)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("no MP4 encoder on this system (installing ffmpeg adds one): {e}"))?;
        Ok(Box::new(Self { child, next: 0.0, last: Vec::new() }))
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
impl VideoSink for CliSink {
    fn push(&mut self, rgba: &[u8], t: f64) -> Result<(), String> {
        let stdin = self.child.stdin.as_mut().ok_or("ffmpeg closed its input")?;
        // Constant rate: repeat the last frame across a long one.
        while !self.last.is_empty() && self.next + 1.0 / FPS <= t {
            stdin.write_all(&self.last).map_err(|e| e.to_string())?;
            self.next += 1.0 / FPS;
        }
        stdin.write_all(rgba).map_err(|e| e.to_string())?;
        self.next += 1.0 / FPS;
        self.last = rgba.to_vec();
        Ok(())
    }
    fn finish(mut self: Box<Self>, _end: f64) -> Result<(), String> {
        drop(self.child.stdin.take());
        let st = self.child.wait().map_err(|e| e.to_string())?;
        if st.success() { Ok(()) } else { Err(format!("ffmpeg failed ({st})")) }
    }
}

/// Encode frames `frames` of the take: an MP4 (and a GIF if asked) into
/// `dir` (the save directory).
#[allow(clippy::too_many_arguments)]
fn save(
    dir: &Path,
    file: &Path,
    frames: &[FrameRec],
    w: u32,
    h: u32,
    px_per_pt: f32,
    opts: &Opts,
    done: &AtomicUsize,
    cancel: &AtomicBool,
) -> Result<Saved, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot make {}: {e}", dir.display()))?;
    let stem = share::stamp_name();
    let mp4_path = dir.join(format!("{stem}.mp4"));
    let gif_path = dir.join(format!("{stem}.gif"));
    let mut note = None;
    let mut mp4 = match open_mp4(&mp4_path, w, h) {
        Ok(s) => Some(s),
        Err(e) => {
            note = Some(e);
            None
        }
    };
    let mut gif = if opts.gif || mp4.is_none() { Some(GifOut::new(&gif_path, w, h)?) } else { None };
    let mut f = std::fs::File::open(file).map_err(|e| e.to_string())?;
    let t0 = frames.first().map(|f| f.t).unwrap_or(0.0);
    let mut buf = Vec::new();
    for rec in frames {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        f.seek(SeekFrom::Start(rec.offset)).map_err(|e| e.to_string())?;
        buf.resize(rec.len as usize, 0);
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        let mut px = lz4_flex::decompress_size_prepended(&buf).map_err(|e| e.to_string())?;
        overlay::apply(&mut px, w, h, rec.cursor.filter(|_| opts.cursor), px_per_pt, opts.watermark);
        let t = rec.t - t0;
        if let Some(m) = mp4.as_mut() {
            m.push(&px, t)?;
        }
        if let Some(g) = gif.as_mut() {
            g.push(&px, t)?;
        }
        done.fetch_add(1, Ordering::Release);
    }
    let end = frames.last().map(|f| f.t - t0).unwrap_or(0.0) + 1.0 / FPS;
    if let Some(m) = mp4 {
        m.finish(end)?;
    }
    if let Some(g) = gif {
        g.finish(end)?;
    }
    Ok(Saved {
        mp4: mp4_path.exists().then_some(mp4_path),
        gif: gif_path.exists().then_some(gif_path),
        note,
    })
}

/// GIF frames at [`GIF_FPS`], shrunk to [`GIF_MAX_W`]; each frame is held
/// until the next one arrives so its delay is the real gap.
struct GifOut {
    enc: gif::Encoder<std::io::BufWriter<std::fs::File>>,
    w: u32,
    h: u32,
    k: u32,
    gw: u16,
    gh: u16,
    next: f64,
    pending: Option<(Vec<u8>, f64)>,
}

impl GifOut {
    fn new(path: &Path, w: u32, h: u32) -> Result<Self, String> {
        let k = w.div_ceil(GIF_MAX_W).max(1);
        let (gw, gh) = ((w / k) as u16, (h / k) as u16);
        let file = std::fs::File::create(path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        let mut enc = gif::Encoder::new(std::io::BufWriter::new(file), gw, gh, &[]).map_err(|e| e.to_string())?;
        enc.set_repeat(gif::Repeat::Infinite).map_err(|e| e.to_string())?;
        Ok(Self { enc, w, h, k, gw, gh, next: 0.0, pending: None })
    }

    fn push(&mut self, rgba: &[u8], t: f64) -> Result<(), String> {
        if t + 1e-6 < self.next {
            return Ok(());
        }
        self.next = t + 1.0 / GIF_FPS;
        let small = box_down(rgba, self.w, self.h, self.k);
        if let Some((px, t0)) = self.pending.replace((small, t)) {
            self.write(px, t - t0)?;
        }
        Ok(())
    }

    fn write(&mut self, mut px: Vec<u8>, dur: f64) -> Result<(), String> {
        let mut frame = gif::Frame::from_rgba_speed(self.gw, self.gh, &mut px, 10);
        frame.delay = ((dur * 100.0).round() as u16).max(2);
        self.enc.write_frame(&frame).map_err(|e| e.to_string())
    }

    fn finish(mut self, end: f64) -> Result<(), String> {
        if let Some((px, t0)) = self.pending.take() {
            self.write(px, (end - t0).max(1.0 / GIF_FPS))?;
        }
        Ok(())
    }
}

/// Box downscale of RGBA by a whole factor (size rounded down).
fn box_down(rgba: &[u8], w: u32, h: u32, k: u32) -> Vec<u8> {
    if k == 1 {
        return rgba.to_vec();
    }
    let (ow, oh, k, w) = ((w / k) as usize, (h / k) as usize, k as usize, w as usize);
    let mut out = vec![0u8; ow * oh * 4];
    let n = (k * k) as u32;
    for y in 0..oh {
        for x in 0..ow {
            for c in 0..4 {
                let mut acc = 0u32;
                for dy in 0..k {
                    for dx in 0..k {
                        acc += rgba[((y * k + dy) * w + x * k + dx) * 4 + c] as u32;
                    }
                }
                out[(y * ow + x) * 4 + c] = ((acc + n / 2) / n) as u8;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_fits_the_cap_with_even_sides() {
        // A Retina window: halved, and still even.
        let g = Geometry::new([0.0, 0.0, 3456.0, 2234.0], 3456, 2234).unwrap();
        assert_eq!(g.k, 2);
        assert_eq!((g.out_w, g.out_h), (1728, 1116));
        assert_eq!((g.w, g.h), (g.out_w * 2, g.out_h * 2));
        // An odd crop rounds down to even, at 1:1.
        let g = Geometry::new([10.4, 20.0, 801.0, 601.0], 2000, 2000).unwrap();
        assert_eq!(g.k, 1);
        assert_eq!((g.x, g.y, g.out_w, g.out_h), (10, 20, 800, 600));
        // Clamped into the surface; nothing left is None.
        let g = Geometry::new([100.0, 100.0, 5000.0, 5000.0], 1000, 800).unwrap();
        assert!(g.x + g.w <= 1000 && g.y + g.h <= 800);
        assert!(Geometry::new([990.0, 0.0, 100.0, 100.0], 1000, 800).is_none());
    }

    #[test]
    fn shrink_averages_and_swizzles() {
        let geo = Geometry { x: 0, y: 0, w: 4, h: 2, k: 2, out_w: 2, out_h: 1 };
        // BGRA, rows padded to 256 bytes.
        let mut data = vec![0u8; 256 * 2];
        for y in 0..2 {
            for x in 0..4 {
                let p = &mut data[y * 256 + x * 4..][..4];
                p.copy_from_slice(&[10 * x as u8, 100, 200, 255]); // b g r a
            }
        }
        let raw = RawFrame { data, row_bytes: 256, bgra: true, t: 0.0, cursor: None };
        let out = shrink(&raw, &geo);
        // Left pixel: b = mean(0,10) = 5 -> lands in the blue slot.
        assert_eq!(&out[0..4], &[200, 100, 5, 255]);
        assert_eq!(&out[4..8], &[200, 100, 25, 255]);
    }

    #[test]
    fn a_take_round_trips_and_saves_a_gif() {
        let geo = Geometry::new([0.0, 0.0, 64.0, 48.0], 64, 48).unwrap();
        let dir = std::env::temp_dir().join(format!("wobble-rec-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("frames.lz4");
        let shared = Arc::new(Mutex::new(TakeShared::default()));
        let (tx, rx) = mpsc::channel();
        let out = std::fs::File::create(&file).unwrap();
        let s2 = shared.clone();
        let h = std::thread::spawn(move || worker(rx, out, geo, s2));
        for i in 0..6u8 {
            let data = vec![i * 40; 256 * 48];
            tx.send(Some(RawFrame { data, row_bytes: 256, bgra: false, t: i as f64 / FPS, cursor: Some([10.0, 10.0]) }))
                .unwrap();
        }
        tx.send(None).unwrap();
        h.join().unwrap();
        let take = Take { dir: dir.clone(), file, w: geo.out_w, h: geo.out_h, px_per_pt: 1.0, shared };
        assert_eq!(take.shared.lock().unwrap().frames.len(), 6);
        let px = take.read_frame(3).unwrap();
        assert_eq!(px.len(), (64 * 48 * 4) as usize);
        assert_eq!(px[0], 120);
        let frames = take.shared.lock().unwrap().frames.clone();
        let gif_path = dir.join("t.gif");
        let mut g = GifOut::new(&gif_path, 64, 48).unwrap();
        for f in &frames {
            let mut px = take.read_frame(frames.iter().position(|x| x.offset == f.offset).unwrap()).unwrap();
            overlay::apply(&mut px, 64, 48, f.cursor, 1.0, true);
            g.push(&px, f.t).unwrap();
        }
        g.finish(6.0 / FPS).unwrap();
        let bytes = std::fs::read(&gif_path).unwrap();
        assert_eq!(&bytes[..6], b"GIF89a");
    }

    /// The platform encoder end to end: two seconds of a moving bar at the
    /// cap size, an odd-but-even size, and a small one, each an mp4 with
    /// frames in it. `EDIFY_REC_KEEP=1` leaves them in the temp dir.
    #[test]
    #[cfg(any(target_os = "macos", target_os = "windows"))]
    fn the_platform_encoder_writes_an_mp4() {
        let dir = std::env::temp_dir().join(format!("wobble-mp4-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (w, h) in [(MAX_W, 1080), (1278, 718), (320, 240)] {
            let path = dir.join(format!("t{w}x{h}.mp4"));
            let mut sink = open_mp4(&path, w, h).unwrap();
            let n = (2.0 * FPS) as u32;
            for i in 0..n {
                let mut px = vec![0u8; (w * h * 4) as usize];
                let bar = (i * w / n) as usize;
                for (j, p) in px.chunks_exact_mut(4).enumerate() {
                    let x = j % w as usize;
                    let on = x >= bar && x < bar + (w as usize / 10);
                    p.copy_from_slice(&if on { [250, 40, 40, 255] } else { [20, 20, 30, 255] });
                }
                sink.push(&px, i as f64 / FPS).unwrap();
            }
            sink.finish(n as f64 / FPS).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(&bytes[4..8], b"ftyp", "{w}x{h}: not an mp4");
            assert!(bytes.windows(4).any(|b| b == b"moov"), "{w}x{h}: no moov");
            assert!(bytes.len() > 4000, "{w}x{h}: only {} bytes", bytes.len());
            eprintln!("{w}x{h}: {} bytes -> {}", bytes.len(), path.display());
        }
        if media::sys::var_os("REC_KEEP").is_none() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
