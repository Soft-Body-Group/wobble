//! Poster frames for the bin: one small RGBA8 thumbnail per clip, decoded
//! on a worker from the clip's middle frame through the engine's own
//! decoders, tone-mapped for display, uploaded as an egui texture on the
//! next frame.

use engine::decode::ExrDecoder;
use half::f16;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;

/// Longest side of a thumbnail, in pixels.
pub const THUMB_PX: u32 = 192;

pub enum Thumb {
    Loading,
    Failed(String),
    Ready(egui::TextureHandle),
}

struct Req {
    uid: u64,
    path: PathBuf,
}

struct Done {
    uid: u64,
    result: Result<(u32, u32, Vec<u8>), String>,
}

pub struct Thumbs {
    tx: Sender<Req>,
    rx: Receiver<Done>,
    pub have: HashMap<u64, Thumb>,
    pending: HashSet<u64>,
}

impl Thumbs {
    pub fn new(decoder: Arc<dyn ExrDecoder>) -> Self {
        let (tx, req_rx) = channel::<Req>();
        let (done_tx, rx) = channel::<Done>();
        std::thread::Builder::new()
            .name("bin-thumbs".into())
            .spawn(move || {
                for r in req_rx {
                    let t0 = web_time::Instant::now();
                    let result = decode_thumb(decoder.as_ref(), &r.path).map_err(|e| format!("{e:#}"));
                    log::info!("bin: poster decode for clip {} took {:?}", r.uid, t0.elapsed());
                    if done_tx.send(Done { uid: r.uid, result }).is_err() {
                        break;
                    }
                }
            })
            .expect("thumb thread");
        Self { tx, rx, have: HashMap::new(), pending: HashSet::new() }
    }

    /// Ask for a clip's poster if we have not already.
    pub fn request(&mut self, uid: u64, path: PathBuf) {
        if self.have.contains_key(&uid) || self.pending.contains(&uid) {
            return;
        }
        self.pending.insert(uid);
        self.have.insert(uid, Thumb::Loading);
        log::info!("bin: poster requested for clip {uid}: {}", path.display());
        self.tx.send(Req { uid, path }).ok();
    }

    /// Forget a clip's poster (the media changed).
    pub fn invalidate(&mut self, uid: u64) {
        self.have.remove(&uid);
        self.pending.remove(&uid);
    }

    /// Upload finished decodes. Call once per frame from a UI pass.
    pub fn poll(&mut self, ctx: &egui::Context) {
        while let Ok(d) = self.rx.try_recv() {
            self.pending.remove(&d.uid);
            match &d.result {
                Ok((w, h, _)) => log::info!("bin: poster for clip {} ready ({w}x{h})", d.uid),
                Err(e) => log::warn!("bin: poster for clip {} failed: {e}", d.uid),
            }
            let t = match d.result {
                Ok((w, h, rgba)) => {
                    let img = egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                    Thumb::Ready(ctx.load_texture(
                        format!("bin-thumb-{}", d.uid),
                        img,
                        egui::TextureOptions::LINEAR,
                    ))
                }
                Err(e) => Thumb::Failed(e),
            };
            self.have.insert(d.uid, t);
            ctx.request_repaint();
        }
    }
}

/// Decode one frame, box-downsample to `THUMB_PX`, tone-map (Narkowicz ACES
/// fit — the same curve the viewer's "approx" transform uses) and encode
/// to sRGB RGBA8.
fn decode_thumb(decoder: &dyn ExrDecoder, path: &std::path::Path) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let info = decoder.read_header(path)?;
    let (w, h) = (info.width as usize, info.height as usize);
    anyhow::ensure!(w > 0 && h > 0, "empty image");
    let stride = w * 4;
    let mut px = vec![f16::ZERO; stride * h];
    decoder.decode_into(path, info, stride, &mut px)?;
    let scale = ((w.max(h) as f32) / THUMB_PX as f32).max(1.0);
    let (tw, th) = (((w as f32 / scale) as usize).max(1), ((h as f32 / scale) as usize).max(1));
    let mut out = Vec::with_capacity(tw * th * 4);
    for ty in 0..th {
        let y0 = (ty as f32 * scale) as usize;
        let y1 = (((ty + 1) as f32 * scale) as usize).clamp(y0 + 1, h);
        for tx in 0..tw {
            let x0 = (tx as f32 * scale) as usize;
            let x1 = (((tx + 1) as f32 * scale) as usize).clamp(x0 + 1, w);
            let mut acc = [0f32; 3];
            let mut n = 0f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y * stride + x * 4;
                    acc[0] += px[i].to_f32();
                    acc[1] += px[i + 1].to_f32();
                    acc[2] += px[i + 2].to_f32();
                    n += 1.0;
                }
            }
            for c in acc.iter_mut() {
                *c /= n.max(1.0);
            }
            for c in acc {
                let v = tone(c.max(0.0));
                out.push((srgb(v) * 255.0 + 0.5) as u8);
            }
            out.push(255);
        }
    }
    Ok((tw as u32, th as u32, out))
}

fn tone(x: f32) -> f32 {
    // Narkowicz 2015 ACES fit.
    let (a, b, c, d, e) = (2.51, 0.03, 2.43, 0.59, 0.14);
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

fn srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}
