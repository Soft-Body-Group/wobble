//! "Submit to the wobble blog": a clip from the recorder, or what a Write
//! rendered, sent to the site's moderation queue (wobble.ing/submit/'s own
//! endpoint) with a title, a description, an optional name and, if the user
//! ticks it, the current setup as a `.wbl`.
//!
//! Nothing is sent without the dialog's button, and nothing shows on the
//! site until it is approved there; the dialog says so. The site takes a
//! video (mp4/webm/mov, 95 MB) or an image (png/jpeg/gif/webp, 20 MB), so a
//! Write's EXRs are first turned into what the viewer shows — through the
//! view transform, box-downscaled to fit 1920x1200 — and encoded with the
//! recorder's encoder (an MP4), or written as a PNG when it is one frame.
//!
//! The upload is the site's four steps (`website_wobble/src/wobbles.ts`):
//! create (words + sizes, answering an id and a one-off key), PUT the media,
//! PUT the .wbl, mark done. Its spam guard applies to us too: `t` is when
//! the dialog opened and the button stays off for the site's minimum.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use web_time::Instant;

pub const SITE: &str = "https://wobble.ing";
/// The site's limits (`LIMITS` in wobbles.ts).
pub const VIDEO_MAX: u64 = 95 * 1024 * 1024;
pub const IMAGE_MAX: u64 = 20 * 1024 * 1024;
pub const WBL_MAX: u64 = 5 * 1024 * 1024;
/// The site's minimum time between opening the form and sending it.
const MIN_FILL: f64 = 3.2;
/// Converted renders are downscaled to fit this, like a recorder take.
const MAX_W: u32 = 1920;
const MAX_H: u32 = 1200;

fn site() -> String {
    std::env::var("WOBBLE_SITE_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| SITE.to_string())
        .trim_end_matches('/')
        .to_string()
}

/// What is being submitted.
#[derive(Clone)]
pub enum Media {
    /// A finished file: a recorder clip.
    File(PathBuf),
    /// A Write's EXR frames, to be shown through `view` (whose `working` is
    /// the files' own primaries) and encoded at `fps`. `curve` is the
    /// files' own, for the rare EXR written with one.
    Frames {
        files: Vec<PathBuf>,
        fps: f32,
        view: engine::viewtf::ViewSettings,
        curve: engine::color::Curve,
    },
}

pub fn mime_of(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "mp4" | "m4v" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => return None,
    })
}

pub fn limit_of(mime: &str) -> u64 {
    if mime.starts_with("video/") { VIDEO_MAX } else { IMAGE_MAX }
}

/// A `.wbl` with this machine's home directory written as `~`: a setup is
/// public once approved, and the paths in it would otherwise name the
/// account it was made on. `expand_home` reads them back on load.
pub fn scrub_home(json: &str) -> String {
    let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
        return json.to_string();
    };
    let home = home.to_string_lossy().trim_end_matches(['/', '\\']).to_string();
    if home.len() < 2 {
        return json.to_string();
    }
    // JSON escapes a backslash, so a Windows home appears doubled.
    json.replace(&home.replace('\\', "\\\\"), "~").replace(&home, "~")
}

struct Job {
    stage: Arc<Mutex<String>>,
    /// Permille of the current stage.
    progress: Arc<AtomicU32>,
    cancel: Arc<AtomicBool>,
    result: Arc<Mutex<Option<Result<(), String>>>>,
}

pub struct Submit {
    media: Media,
    title: String,
    description: String,
    name: String,
    /// The setup as it was when the dialog opened, home scrubbed.
    wbl: Option<String>,
    attach_wbl: bool,
    opened: Instant,
    opened_ms: u64,
    job: Option<Job>,
    done: bool,
    error: Option<String>,
}

impl Submit {
    pub fn new(media: Media, wbl: Option<String>, name: String) -> Self {
        let opened_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            media,
            title: String::new(),
            description: String::new(),
            name,
            wbl: wbl.map(|s| scrub_home(&s)),
            attach_wbl: true,
            opened: Instant::now(),
            opened_ms,
            job: None,
            done: false,
            error: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Draw the dialog. False once it is closed.
    pub fn ui(&mut self, ctx: &egui::Context) -> bool {
        let mut open = true;
        let mut close = false;
        egui::Window::new("submit to the wobble blog")
            .id(egui::Id::new("blog-submit"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(380.0)
            .default_pos(ctx.content_rect().center() - egui::vec2(190.0, 160.0))
            .show(ctx, |ui| close = self.body(ui));
        if !open || close {
            if let Some(j) = &self.job {
                j.cancel.store(true, Ordering::Release);
            }
            return false;
        }
        true
    }

    fn body(&mut self, ui: &mut egui::Ui) -> bool {
        if self.done {
            ui.label("Sent. It shows on wobble.ing/wobbles/ once it has been approved.");
            ui.add_space(4.0);
            return ui.button("close").clicked();
        }
        let busy = self.job.is_some();
        ui.label(match &self.media {
            Media::File(p) => {
                let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
                format!("{} · {:.1} MB", p.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(), size as f64 / 1e6)
            }
            Media::Frames { files, .. } if files.len() == 1 => "the rendered frame, as a PNG".into(),
            Media::Frames { files, .. } => format!("the {} rendered frames, as an MP4", files.len()),
        });
        if let Media::Frames { view, .. } = &self.media {
            if view.display_rgb([0.18; 3]).is_none() {
                ui.weak("OpenDRT has no CPU version, so this is encoded through Filmic");
            }
        }
        ui.add_space(4.0);
        ui.add_enabled_ui(!busy, |ui| {
            egui::Grid::new("blog-fields").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label("title");
                ui.add(egui::TextEdit::singleline(&mut self.title).char_limit(120).desired_width(280.0));
                ui.end_row();
                ui.label("about it");
                ui.add(
                    egui::TextEdit::multiline(&mut self.description)
                        .char_limit(4000)
                        .desired_rows(4)
                        .desired_width(280.0)
                        .hint_text("optional: how it was made"),
                );
                ui.end_row();
                ui.label("your name");
                ui.add(egui::TextEdit::singleline(&mut self.name).char_limit(80).desired_width(280.0).hint_text("optional"));
                ui.end_row();
            });
            match &self.wbl {
                Some(w) if (w.len() as u64) <= WBL_MAX => {
                    ui.checkbox(&mut self.attach_wbl, "attach this comp's setup (.wbl)")
                        .on_hover_text("so others can open it; file paths keep only what is under your home folder, as ~");
                }
                Some(_) => {
                    ui.weak("the setup is over 5 MB, too big to attach");
                }
                None => {}
            }
        });
        ui.add_space(6.0);
        if let Some(j) = &self.job {
            let stage = j.stage.lock().map(|s| s.clone()).unwrap_or_default();
            let p = j.progress.load(Ordering::Acquire) as f32 / 1000.0;
            ui.add(egui::ProgressBar::new(p).text(stage));
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            let res = j.result.lock().ok().and_then(|mut r| r.take());
            if let Some(r) = res {
                self.job = None;
                match r {
                    Ok(()) => self.done = true,
                    Err(e) => self.error = Some(e),
                }
            } else if ui.button("cancel").clicked() {
                j.cancel.store(true, Ordering::Release);
            }
        } else {
            let wait = MIN_FILL - self.opened.elapsed().as_secs_f64();
            if wait > 0.0 {
                ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(wait));
            }
            let ready = self.title.trim().chars().count() >= 2 && wait <= 0.0;
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(ready, egui::Button::new("send"))
                    .on_disabled_hover_text("give it a title")
                    .clicked()
                {
                    self.error = None;
                    self.job = Some(self.start());
                }
                ui.weak("approved by hand before it appears on the site");
            });
        }
        if let Some(e) = &self.error {
            ui.colored_label(crate::theme::error(), e);
        }
        false
    }

    fn start(&self) -> Job {
        let job = Job {
            stage: Arc::new(Mutex::new("preparing".into())),
            progress: Arc::new(AtomicU32::new(0)),
            cancel: Arc::new(AtomicBool::new(false)),
            result: Arc::new(Mutex::new(None)),
        };
        let post = Post {
            title: self.title.trim().to_string(),
            description: self.description.trim().to_string(),
            name: self.name.trim().to_string(),
            wbl: self.wbl.clone().filter(|w| self.attach_wbl && (w.len() as u64) <= WBL_MAX),
            t: self.opened_ms,
        };
        let media = self.media.clone();
        let (stage, progress, cancel, result) = (job.stage.clone(), job.progress.clone(), job.cancel.clone(), job.result.clone());
        crate::bg::spawn(move || {
            let r = run(media, post, &stage, &progress, &cancel);
            if let Ok(mut s) = result.lock() {
                *s = Some(r);
            }
        });
        job
    }
}

struct Post {
    title: String,
    description: String,
    name: String,
    wbl: Option<String>,
    t: u64,
}

fn run(media: Media, post: Post, stage: &Mutex<String>, progress: &AtomicU32, cancel: &AtomicBool) -> Result<(), String> {
    let set = |s: &str| {
        if let Ok(mut g) = stage.lock() {
            *g = s.to_string();
        }
        progress.store(0, Ordering::Release);
    };
    let mut _tmp = None;
    let path = match media {
        Media::File(p) => p,
        Media::Frames { files, fps, view, curve } => {
            set("converting the render");
            let dir = std::env::temp_dir().join(format!("wobble-blog-{}", std::process::id()));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            _tmp = Some(TempDir(dir.clone()));
            convert(&files, fps, view, curve, &dir, progress, cancel)?
        }
    };
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let mime = mime_of(&path).ok_or("the blog takes mp4, webm, mov, png, jpeg, gif or webp")?;
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let max = limit_of(mime);
    if bytes.len() as u64 > max {
        return Err(format!(
            "{:.0} MB is over the blog's {} MB for {}; trim it shorter",
            bytes.len() as f64 / 1e6,
            max / (1024 * 1024),
            if mime.starts_with("video/") { "a video" } else { "an image" }
        ));
    }
    set("sending");
    upload(&post, mime, &bytes, progress, cancel)
}

/// Removed with its contents when the job ends.
struct TempDir(PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One rendered EXR, as display-referred 8-bit RGBA at the take size.
fn display_frame(
    path: &Path,
    view: &engine::viewtf::ViewSettings,
    curve: engine::color::Curve,
) -> Result<(u32, u32, Vec<u8>), String> {
    use exr::prelude::*;
    use rayon::prelude::*;
    let img = read_first_rgba_layer_from_file(
        path,
        |res, _| (res.width(), res.height(), vec![[0f32; 3]; res.width() * res.height()]),
        |(w, _, px): &mut (usize, usize, Vec<[f32; 3]>), pos, (r, g, b, _a): (f32, f32, f32, f32)| {
            px[pos.y() * *w + pos.x()] = [r, g, b];
        },
    )
    .map_err(|e| format!("{}: {e}", path.display()))?;
    let (w, h, px) = img.layer_data.channel_data.pixels;
    let (ow, oh, k) = fit(w as u32, h as u32);
    let (ow, oh, k) = (ow as usize, oh as usize, k as usize);
    let n = (k * k) as f32;
    let mut out = vec![255u8; ow * oh * 4];
    out.par_chunks_mut(ow * 4).enumerate().for_each(|(y, row)| {
        for x in 0..ow {
            let mut acc = [0f32; 3];
            for dy in 0..k {
                for dx in 0..k {
                    let p = px[(y * k + dy) * w + x * k + dx];
                    for c in 0..3 {
                        let v = if p[c].is_finite() { p[c] } else { 0.0 };
                        acc[c] += curve.to_linear(v);
                    }
                }
            }
            let lin = acc.map(|v| v / n);
            let d = view.display_rgb(lin).unwrap_or([0.0; 3]);
            for c in 0..3 {
                row[x * 4 + c] = (d[c].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    });
    Ok((ow as u32, oh as u32, out))
}

/// The output size for a `w`x`h` render: a whole box factor that fits
/// [`MAX_W`]x[`MAX_H`], and even sides (H.264 wants them).
fn fit(w: u32, h: u32) -> (u32, u32, u32) {
    let k = w.div_ceil(MAX_W).max(h.div_ceil(MAX_H)).max(1);
    ((w / k).max(2) & !1, (h / k).max(2) & !1, k)
}

fn convert(
    files: &[PathBuf],
    fps: f32,
    mut view: engine::viewtf::ViewSettings,
    curve: engine::color::Curve,
    dir: &Path,
    progress: &AtomicU32,
    cancel: &AtomicBool,
) -> Result<PathBuf, String> {
    if view.display_rgb([0.18; 3]).is_none() {
        view.transform = engine::viewtf::ViewTransform::Filmic;
    }
    let first = files.first().ok_or("nothing was rendered")?;
    if files.len() == 1 {
        let (w, h, rgba) = display_frame(first, &view, curve)?;
        let path = dir.join("wobble.png");
        write_png(&path, w, h, &rgba)?;
        return Ok(path);
    }
    let (w, h, rgba) = display_frame(first, &view, curve)?;
    let path = dir.join("wobble.mp4");
    let mut sink = crate::recorder::open_mp4(&path, w, h)?;
    let dt = 1.0 / fps.max(1.0) as f64;
    sink.push(&rgba, 0.0)?;
    for (i, f) in files.iter().enumerate().skip(1) {
        if cancel.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        let (fw, fh, rgba) = display_frame(f, &view, curve)?;
        if (fw, fh) != (w, h) {
            return Err(format!("{} is a different size from the first frame", f.display()));
        }
        sink.push(&rgba, i as f64 * dt)?;
        progress.store((i * 1000 / files.len()) as u32, Ordering::Release);
    }
    sink.finish(files.len() as f64 * dt)?;
    Ok(path)
}

fn write_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    let f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut wr = enc.write_header().map_err(|e| e.to_string())?;
    wr.write_image_data(rgba).map_err(|e| e.to_string())
}

/// What the site's error codes mean to the person who pressed send.
fn explain(status: u16, body: &str) -> String {
    let code = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
        .unwrap_or_default();
    match code.as_str() {
        "slow_down" => "the blog has had enough from here for today; try again tomorrow".into(),
        "queue_full" => "the blog's queue is full right now; try again later".into(),
        "too_large" => "the file is over the blog's size limit".into(),
        "wbl_too_large" => "the setup is over 5 MB".into(),
        "no_title" => "give it a title".into(),
        "bad_type" => "the site did not accept this kind of file".into(),
        "closed" | "not_found" => "the upload timed out; send it again".into(),
        _ => format!("the site said {status}{}", if code.is_empty() { String::new() } else { format!(" ({code})") }),
    }
}

#[cfg(target_arch = "wasm32")]
fn upload(_post: &Post, _mime: &str, _bytes: &[u8], _progress: &AtomicU32, _cancel: &AtomicBool) -> Result<(), String> {
    Err("the web build cannot upload".into())
}

#[cfg(not(target_arch = "wasm32"))]
fn upload(post: &Post, mime: &str, bytes: &[u8], progress: &AtomicU32, cancel: &AtomicBool) -> Result<(), String> {
    use std::time::Duration;
    let site = site();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(600)))
        .http_status_as_error(false)
        .user_agent(format!("wobble/{}", env!("CARGO_PKG_VERSION")))
        .build()
        .into();
    let check = |mut r: ureq::http::Response<ureq::Body>| -> Result<String, String> {
        let status = r.status().as_u16();
        let text = r.body_mut().read_to_string().unwrap_or_default();
        if (200..300).contains(&status) { Ok(text) } else { Err(explain(status, &text)) }
    };
    let body = serde_json::json!({
        "title": post.title,
        "description": post.description,
        "name": post.name,
        "media": { "type": mime, "size": bytes.len() },
        "wbl": post.wbl.as_ref().map(|w| serde_json::json!({ "size": w.len() })),
        "website": "",
        "t": post.t,
    });
    let created = check(
        agent
            .post(format!("{site}/api/wobbles"))
            .header("origin", &site)
            .header("content-type", "application/json")
            .send(body.to_string())
            .map_err(|e| format!("could not reach {site}: {e}"))?,
    )?;
    let v: serde_json::Value = serde_json::from_str(&created).map_err(|e| e.to_string())?;
    let (Some(id), Some(key)) = (v["id"].as_str(), v["key"].as_str()) else {
        return Err("the site's answer made no sense".into());
    };
    progress.store(100, Ordering::Release);
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    let put = |what: &str, ty: &str, data: &[u8]| -> Result<String, String> {
        check(
            agent
                .put(format!("{site}/api/wobbles/{id}/{what}"))
                .header("origin", &site)
                .header("x-upload-key", key)
                .header("content-type", ty)
                .send(data)
                .map_err(|e| format!("upload failed: {e}"))?,
        )
    };
    put("media", mime, bytes)?;
    progress.store(900, Ordering::Release);
    if let Some(w) = &post.wbl {
        put("wbl", "application/json", w.as_bytes())?;
    }
    if cancel.load(Ordering::Acquire) {
        return Err("cancelled".into());
    }
    check(
        agent
            .post(format!("{site}/api/wobbles/{id}/done"))
            .header("origin", &site)
            .header("x-upload-key", key)
            .send_empty()
            .map_err(|e| format!("could not reach {site}: {e}"))?,
    )?;
    progress.store(1000, Ordering::Release);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_keeps_even_sides_under_the_cap() {
        assert_eq!(fit(1920, 1080), (1920, 1080, 1));
        assert_eq!(fit(4096, 2160), (1364, 720, 3));
        assert_eq!(fit(2048, 1080), (1024, 540, 2));
        let (w, h, _) = fit(1001, 333);
        assert!(w % 2 == 0 && h % 2 == 0);
    }

    #[test]
    fn mimes_match_the_site() {
        assert_eq!(mime_of(Path::new("a/b.MP4")), Some("video/mp4"));
        assert_eq!(mime_of(Path::new("x.gif")), Some("image/gif"));
        assert_eq!(mime_of(Path::new("x.exr")), None);
        assert_eq!(limit_of("image/gif"), IMAGE_MAX);
        assert_eq!(limit_of("video/mp4"), VIDEO_MAX);
    }

    #[test]
    fn the_home_dir_leaves_a_setup() {
        let Some(home) = std::env::var_os("HOME") else { return };
        let home = home.to_string_lossy().into_owned();
        let json = format!("{{\"path\":\"{home}/shots/a.####.exr\"}}");
        let s = scrub_home(&json);
        assert_eq!(s, "{\"path\":\"~/shots/a.####.exr\"}");
    }

    #[test]
    fn the_site_errors_read_as_words() {
        assert!(explain(429, r#"{"error":"slow_down"}"#).contains("tomorrow"));
        assert!(explain(500, "boom").contains("500"));
    }

    /// A render becomes a PNG at the take size through the view transform.
    #[test]
    fn a_one_frame_render_becomes_a_png() {
        use exr::prelude::*;
        let dir = std::env::temp_dir().join(format!("wobble-blog-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let exr = dir.join("f.0001.exr");
        write_rgba_file(&exr, 64, 32, |x, _| (x as f32 / 64.0, 0.18f32, 0.0f32, 1.0f32)).unwrap();
        let view = engine::viewtf::ViewSettings::default();
        let out = convert(&[exr], 24.0, view, engine::color::Curve::Linear, &dir, &AtomicU32::new(0), &AtomicBool::new(false)).unwrap();
        assert_eq!(mime_of(&out), Some("image/png"));
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&out).unwrap()));
        let r = dec.read_info().unwrap();
        assert_eq!((r.info().width, r.info().height), (64, 32));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Several frames become an MP4 through the platform encoder.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_sequence_becomes_an_mp4() {
        use exr::prelude::*;
        let dir = std::env::temp_dir().join(format!("wobble-blog-seq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let files: Vec<_> = (0..6)
            .map(|i| {
                let p = dir.join(format!("f.{i:04}.exr"));
                write_rgba_file(&p, 2048, 1080, |x, _| ((x + i * 40) as f32 / 2048.0, 0.18f32, 0.05f32, 1.0f32)).unwrap();
                p
            })
            .collect();
        let view = engine::viewtf::ViewSettings::default();
        let out = convert(&files, 24.0, view, engine::color::Curve::Linear, &dir, &AtomicU32::new(0), &AtomicBool::new(false)).unwrap();
        assert_eq!(mime_of(&out), Some("video/mp4"));
        assert!(std::fs::metadata(&out).unwrap().len() > 1000);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The whole upload against a running site: `WOBBLE_SITE_URL` (a
    /// `wrangler dev` of website_wobble with its D1 migrated locally).
    /// Sends a PNG and a .wbl and expects the queue to take them.
    #[test]
    #[ignore]
    fn live_upload() {
        assert!(std::env::var("WOBBLE_SITE_URL").is_ok(), "set WOBBLE_SITE_URL");
        let dir = std::env::temp_dir().join(format!("wobble-blog-live-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("card.png");
        write_png(&png, 4, 2, &[200u8; 32]).unwrap();
        let post = Post {
            title: "live upload test".into(),
            description: "from blogpost::tests::live_upload".into(),
            name: String::new(),
            wbl: Some(scrub_home(r#"{"nodes":[]}"#)),
            t: 1,
        };
        let r = run(Media::File(png), post, &Mutex::new(String::new()), &AtomicU32::new(0), &AtomicBool::new(false));
        let _ = std::fs::remove_dir_all(&dir);
        r.unwrap();
    }
}
