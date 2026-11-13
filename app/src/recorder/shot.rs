//! The screenshot button: one PNG of the wobble window as wobble drew it.
//!
//! Same source as the recorder: the finished surface texture (UI and all)
//! is copied to a readback buffer right before the submit that draws it,
//! so the OS title bar and window chrome are never in it and no
//! screen-recording permission is asked for. A click arms it; the NEXT
//! redraw (the one that can turn `COPY_SRC` on before acquiring its frame)
//! takes the copy; the PNG is written off the UI thread.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use web_time::Instant;

/// How long the "saved" note stays in the menubar.
const NOTE_SECS: f64 = 6.0;

struct Inflight {
    buf: wgpu::Buffer,
    /// 0 copy recorded, 1 mapping, 2 mapped, 3 failed.
    state: Arc<AtomicU32>,
    w: u32,
    h: u32,
    row_bytes: u32,
    bgra: bool,
}

#[derive(Default)]
pub struct Shot {
    /// Clicked: the next redraw takes the picture.
    pending: bool,
    inflight: Option<Inflight>,
    /// The writer thread's answer.
    result: Arc<Mutex<Option<Result<PathBuf, String>>>>,
    /// What the menubar says, and since when.
    pub note: Option<(Result<PathBuf, String>, Instant)>,
}

/// Where screenshots are saved: `WOBBLE_RECORD_DIR`, else the folder
/// picked in the recorder's options, else `default_dir`.
pub fn save_dir() -> PathBuf {
    super::picked_dir(true).unwrap_or_else(default_dir)
}

/// `~/Pictures/Wobble`.
pub fn default_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    home.join("Pictures").join("Wobble")
}

impl Shot {
    /// Take a picture on the next redraw.
    pub fn request(&mut self) {
        if self.inflight.is_none() {
            self.pending = true;
        }
    }

    /// The surface must be copyable on this redraw.
    pub fn wants_copy(&self) -> bool {
        self.pending
    }

    /// Busy from the click to the file landing.
    pub fn busy(&self) -> bool {
        self.pending || self.inflight.is_some()
    }

    /// Record the copy of the finished frame. Call with the encoder that
    /// draws `texture`, after everything that draws into it.
    pub fn capture(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        format: wgpu::TextureFormat,
    ) {
        if !self.pending {
            return;
        }
        self.pending = false;
        let bgra = match format {
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => true,
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => false,
            f => {
                self.note = Some((Err(format!("cannot save a {f:?} window as a PNG")), Instant::now()));
                return;
            }
        };
        let size = texture.size();
        let (w, h) = (size.width, size.height);
        let row_bytes = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot-readback"),
            size: row_bytes as u64 * h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row_bytes), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.inflight = Some(Inflight { buf, state: Arc::new(AtomicU32::new(0)), w, h, row_bytes, bgra });
    }

    /// After the submit that carried the copy.
    pub fn after_submit(&mut self) {
        let Some(f) = &self.inflight else { return };
        if f.state.load(Ordering::Acquire) != 0 {
            return;
        }
        f.state.store(1, Ordering::Release);
        let state = f.state.clone();
        f.buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { 2 } else { 3 }, Ordering::Release);
        });
    }

    /// Once a frame: hand a landed readback to the writer, pick up its
    /// answer. True while there is anything to wait for (keep redrawing).
    pub fn poll(&mut self, device: &wgpu::Device) -> bool {
        if let Some(r) = self.result.lock().ok().and_then(|mut g| g.take()) {
            self.note = Some((r, Instant::now()));
        }
        let Some(f) = &self.inflight else { return self.pending };
        let _ = device.poll(wgpu::PollType::Poll);
        match f.state.load(Ordering::Acquire) {
            2 => {}
            3 => {
                self.inflight = None;
                self.note = Some((Err("the GPU readback failed".into()), Instant::now()));
                return false;
            }
            _ => return true,
        }
        let f = self.inflight.take().expect("checked above");
        let mut rgb = Vec::with_capacity((f.w * f.h * 3) as usize);
        {
            let Ok(data) = f.buf.slice(..).get_mapped_range() else {
                self.note = Some((Err("the GPU readback failed".into()), Instant::now()));
                return false;
            };
            for row in data.chunks_exact(f.row_bytes as usize).take(f.h as usize) {
                for px in row[..(f.w * 4) as usize].chunks_exact(4) {
                    // Opaque RGB: the window has no meaningful alpha.
                    if f.bgra {
                        rgb.extend_from_slice(&[px[2], px[1], px[0]]);
                    } else {
                        rgb.extend_from_slice(&px[..3]);
                    }
                }
            }
        }
        f.buf.unmap();
        let (w, h) = (f.w, f.h);
        let result = self.result.clone();
        let job = move || {
            let r = write_png(w, h, &rgb);
            if let Ok(mut g) = result.lock() {
                *g = Some(r);
            }
        };
        if let Err(e) = std::thread::Builder::new().name("screenshot".into()).spawn(job) {
            self.note = Some((Err(format!("cannot start the writer: {e}")), Instant::now()));
            return false;
        }
        true
    }

    /// The note to show, while it is fresh.
    pub fn fresh_note(&self) -> Option<&Result<PathBuf, String>> {
        self.note.as_ref().filter(|(_, at)| at.elapsed().as_secs_f64() < NOTE_SECS).map(|(r, _)| r)
    }

    /// The menubar control, beside the recorder's. True when clicked.
    pub fn bar_button(&self, ui: &mut egui::Ui) -> bool {
        let size = egui::vec2(22.0, ui.spacing().interact_size.y);
        let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
        let vis = ui.style().interact(&resp);
        ui.painter().rect_filled(rect, 0.0, vis.weak_bg_fill);
        // A little camera: body, lens, the bump on top.
        let c = rect.center();
        let ink = vis.text_color();
        let stroke = egui::Stroke::new(1.2, ink);
        let body = egui::Rect::from_center_size(c + egui::vec2(0.0, 1.0), egui::vec2(13.0, 9.0));
        ui.painter().rect_stroke(body, 1.5, stroke, egui::StrokeKind::Middle);
        ui.painter().circle_stroke(body.center(), 2.6, stroke);
        let bump = egui::Rect::from_min_size(egui::pos2(body.min.x + 3.0, body.min.y - 2.0), egui::vec2(4.0, 2.0));
        ui.painter().rect_filled(bump, 0.0, ink);
        // No tooltip on the redraw that takes the picture: it would be in it.
        let resp = if self.busy() {
            resp
        } else {
            resp.on_hover_text(format!("save a PNG of the wobble window\nto {}", save_dir().display()))
        };
        resp.clicked()
    }
}

fn write_png(w: u32, h: u32, rgb: &[u8]) -> Result<PathBuf, String> {
    let dir = save_dir();
    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot make {}: {e}", dir.display()))?;
    let stem = super::share::stamp_name();
    let mut path = dir.join(format!("{stem}.png"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stem}-{n}.png"));
        n += 1;
    }
    let f = std::fs::File::create(&path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(rgb).map_err(|e| e.to_string())?;
    wr.finish().map_err(|e| e.to_string())?;
    Ok(path)
}
