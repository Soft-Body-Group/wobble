//! Export: the sequence (or its in..out) rendered to an H.264 + AAC MP4 for
//! YouTube or Instagram, exactly as playback shows it: the topmost picture
//! at every frame, speed changes sampled the same way, dissolves mixed,
//! Every audible track's sound mixed by the same `audio::Mixer`.
//!
//! Each picture goes from its decoded frame straight into the output
//! picture in ONE swscale call: the part that will be visible is cropped in
//! the source first (fill mode crops; fit mode letterboxes), then scaled
//! into place, converting its YCbCr matrix and range to Rec.709 video range
//! on the way. A dissolve renders both pictures and blends them. Video is
//! encoded by VideoToolbox (the media engine) where present, else libx264.
//!
//! The file is written under a temporary name and renamed when complete;
//! an existing file is never replaced (the caller picks a free name).

use crate::audio::{plan_of, Mixer};
use anyhow::{bail, Context as _, Result};
use ffmpeg_next as ff;
use media::{Frame, Layout, Matrix, SourceRef};
use model::{Picture, Sequence};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

pub const AUDIO_RATE: u32 = 48_000;
const AUDIO_BITRATE: usize = 192_000;

/// An output format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Preset {
    pub id: &'static str,
    pub name: &'static str,
    pub width: u32,
    pub height: u32,
    /// Video bits per second (per platform guidance, a little above it).
    pub bitrate: usize,
}

/// YouTube's and Instagram's recommended upload shapes.
pub const PRESETS: [Preset; 5] = [
    Preset { id: "youtube-1080", name: "YouTube 1080p 16:9", width: 1920, height: 1080, bitrate: 12_000_000 },
    Preset { id: "youtube-4k", name: "YouTube 4K 16:9", width: 3840, height: 2160, bitrate: 45_000_000 },
    Preset { id: "reels", name: "Instagram Reels / Stories 9:16", width: 1080, height: 1920, bitrate: 10_000_000 },
    Preset { id: "feed-4x5", name: "Instagram feed 4:5", width: 1080, height: 1350, bitrate: 8_000_000 },
    Preset { id: "square", name: "Instagram square 1:1", width: 1080, height: 1080, bitrate: 8_000_000 },
];

pub fn preset(id: &str) -> Option<Preset> {
    PRESETS.iter().copied().find(|p| p.id == id)
}

/// The preset that matches a raster's shape best.
pub fn preset_for(raster: (u32, u32)) -> Preset {
    let a = raster.0 as f64 / raster.1.max(1) as f64;
    let mut best = PRESETS[0];
    for p in PRESETS {
        if p.id == "youtube-4k" {
            continue;
        }
        if (p.width as f64 / p.height as f64 - a).abs() < (best.width as f64 / best.height as f64 - a).abs() {
            best = p;
        }
    }
    best
}

/// How the sequence's raster goes into the output shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// All of it, with black bars.
    Fit,
    /// Fill the output, cropping the overflow (centred).
    Fill,
}

/// A rectangle in output pixels (may run past the output in fill mode).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// Where a `pw`x`ph` picture lands in the output, and where the raster
/// does: the picture placed in the raster by its clip's reframe (exactly as
/// the record viewer places it), the raster fitted or filled into the
/// output. The picture is cut to the raster's area.
pub fn place(pw: u32, ph: u32, raster: (u32, u32), out: (u32, u32), fit: Fit, reframe: model::Reframe) -> (Place, Place) {
    let (rw, rh) = (raster.0.max(1) as f64, raster.1.max(1) as f64);
    let (ow, oh) = (out.0 as f64, out.1 as f64);
    let (px, py, pic_w, pic_h) = reframe.place((pw, ph), raster);
    let s = match fit {
        Fit::Fit => (ow / rw).min(oh / rh),
        Fit::Fill => (ow / rw).max(oh / rh),
    };
    let (ox, oy) = ((ow - rw * s) / 2.0, (oh - rh * s) / 2.0);
    (Place { x: ox + px * s, y: oy + py * s, w: pic_w * s, h: pic_h * s }, Place { x: ox, y: oy, w: rw * s, h: rh * s })
}

/// The output transform of a delivery (every preset is a Rec.709 web
/// upload): the project's view rendered for a Rec.709 display with the sRGB
/// curve, what the viewer shows on an sRGB screen, without the viewer's
/// exposure and gamma (those are for looking, never baked).
pub fn output_view(seq: &Sequence) -> colour::viewtf::ViewSettings {
    let mut v = seq.colour.view;
    v.mirror_working(seq.colour.working);
    v.display = colour::viewtf::Gamut::Rec709;
    v.eotf = colour::viewtf::Eotf::Srgb;
    v.exposure = 0.0;
    v.gamma = 1.0;
    v
}

/// Input transforms by media item, built once a render.
#[derive(Default)]
struct Idts(HashMap<usize, crate::viewer::Idt>);

/// Frame `t`'s picture through the GPU (input transforms, the view, a
/// dissolve's mix), placed in the output as `place` says: RGBA8 rows.
#[allow(clippy::too_many_arguments)]
fn gpu_picture(
    off: &mut crate::viewer::Offscreen,
    idts: &mut Idts,
    seq: &Sequence,
    t: i64,
    raster: (u32, u32),
    out: (u32, u32),
    fit: Fit,
    view: &colour::viewtf::ViewSettings,
    decode: &mut dyn FnMut(usize, u64) -> Option<Frame>,
) -> Vec<u8> {
    let layers: Vec<model::Layer> = match seq.picture_at(t) {
        Picture::Black => vec![],
        Picture::One(l) => vec![l],
        Picture::Mix(a, b, _) => vec![a, b],
    };
    let m = match seq.picture_at(t) {
        Picture::Mix(_, _, m) => m,
        _ => 0.0,
    };
    let frames: Vec<Option<Frame>> = layers.iter().map(|l| decode(l.media, l.frame.max(0) as u64)).collect();
    for l in &layers {
        idts.0.entry(l.media).or_insert_with(|| {
            let c = &seq.colour;
            crate::viewer::Idt::new(c.input_of(&seq.media[l.media].path), c.working)
        });
    }
    let layer = |i: usize| -> Option<crate::viewer::Layer> {
        let f = frames.get(i)?.as_ref()?;
        let l = layers[i];
        let (p, _) = place(f.width, f.height, raster, out, fit, l.reframe);
        let (ow, oh) = (out.0 as f64, out.1 as f64);
        Some(crate::viewer::Layer { frame: f, idt: &idts.0[&l.media], xf: [(p.x / ow) as f32, (p.y / oh) as f32, (p.w / ow) as f32, (p.h / oh) as f32] })
    };
    let (a, b) = (layer(0), layer(1));
    match (a, b) {
        (Some(a), b) => off.frame(Some(a), b, m, view),
        (None, Some(b)) => off.frame(Some(b), None, 0.0, view),
        (None, None) => vec![0; (out.0 * out.1 * 4) as usize],
    }
}

pub struct Job {
    pub seq: Sequence,
    /// The file to read for each sequence media (an optimised copy or the original).
    pub read: Vec<PathBuf>,
    /// Record frames `[start, end)`.
    pub range: (i64, i64),
    pub preset: Preset,
    pub fit: Fit,
    pub out: PathBuf,
    /// Also copy the finished file here (an organisicator result).
    pub copy_to: Option<PathBuf>,
}

#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
}

pub struct Done {
    pub out: PathBuf,
    pub copy: Option<PathBuf>,
    pub frames: u64,
    pub secs: f64,
    pub encoder: &'static str,
}

/// A free name: `path` itself, else `stem_2`, `stem_3`…
pub fn free_name(path: &Path) -> PathBuf {
    if !path.exists() {
        return path.to_path_buf();
    }
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = path.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
    (2..10_000).map(|n| path.with_file_name(format!("{stem}_{n}.{ext}"))).find(|p| !p.exists()).unwrap_or_else(|| path.to_path_buf())
}

/// Render the job. Runs on the caller's thread (start it on a worker).
pub fn run(job: &Job, progress: &Progress) -> Result<Done> {
    media::movie::init();
    let t0 = Instant::now();
    let (a, b) = job.range;
    anyhow::ensure!(b > a, "nothing to render: the range is empty");
    progress.total.store((b - a) as u64, Ordering::Relaxed);
    let (w, h) = (job.preset.width, job.preset.height);
    let raster = job.seq.raster.unwrap_or((w, h));
    let fps = job.seq.fps;

    // Hidden while it is written: organisicator lists a post's out/ and
    // skips dotfiles, so a half-written render never shows as a result.
    let name = job.out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = job.out.with_file_name(format!(".{name}.partial"));
    if let Some(d) = job.out.parent() {
        std::fs::create_dir_all(d)?;
    }
    let r = encode(job, progress, &tmp, (w, h), raster, fps);
    let encoder = match r {
        Ok(e) => e,
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            return Err(e);
        }
    };
    let out = free_name(&job.out);
    std::fs::rename(&tmp, &out)?;
    let copy = match &job.copy_to {
        Some(c) => {
            let c = free_name(c);
            if let Some(d) = c.parent() {
                std::fs::create_dir_all(d)?;
            }
            let cname = c.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let ctmp = c.with_file_name(format!(".{cname}.partial"));
            std::fs::copy(&out, &ctmp).context("copying to the result")?;
            std::fs::rename(&ctmp, &c)?;
            Some(c)
        }
        None => None,
    };
    Ok(Done { out, copy, frames: (b - a) as u64, secs: t0.elapsed().as_secs_f64(), encoder })
}

fn pick_video_encoder() -> Result<(ff::Codec, &'static str)> {
    if media::sys::var("H264").as_deref() != Ok("x264") {
        if let Some(c) = ff::encoder::find_by_name("h264_videotoolbox") {
            return Ok((c, "h264_videotoolbox"));
        }
    }
    let c = ff::encoder::find_by_name("libx264").context("this FFmpeg has no H.264 encoder")?;
    Ok((c, "libx264"))
}

fn encode(job: &Job, progress: &Progress, tmp: &Path, (w, h): (u32, u32), raster: (u32, u32), fps: model::Fps) -> Result<&'static str> {
    let mut octx = ff::format::output_as(tmp, "mp4").with_context(|| format!("creating {}", tmp.display()))?;
    let global = octx.format().flags().contains(ff::format::Flags::GLOBAL_HEADER);

    // Video encoder.
    let (vcodec, vname) = pick_video_encoder()?;
    let vtb = ff::Rational::new(fps.den as i32, fps.num as i32);
    let mut vctx = ff::codec::context::Context::new_with_codec(vcodec);
    vctx.set_time_base(vtb);
    if global {
        vctx.set_flags(ff::codec::Flags::GLOBAL_HEADER);
    }
    let mut venc = vctx.encoder().video()?;
    venc.set_width(w);
    venc.set_height(h);
    venc.set_format(ff::format::Pixel::YUV420P);
    venc.set_bit_rate(job.preset.bitrate);
    venc.set_max_bit_rate(job.preset.bitrate * 3 / 2);
    venc.set_gop((fps.as_f64() * 2.0).round() as u32);
    venc.set_colorspace(ff::color::Space::BT709);
    venc.set_color_range(ff::color::Range::MPEG);
    unsafe {
        let e = venc.as_mut_ptr();
        (*e).framerate = ff::ffi::AVRational { num: fps.num as i32, den: fps.den as i32 };
        (*e).color_primaries = ff::ffi::AVColorPrimaries::AVCOL_PRI_BT709;
        (*e).color_trc = ff::ffi::AVColorTransferCharacteristic::AVCOL_TRC_BT709;
    }
    let mut vopts = ff::Dictionary::new();
    if vname == "h264_videotoolbox" {
        vopts.set("profile", "high");
        vopts.set("allow_sw", "1");
    } else {
        vopts.set("preset", "medium");
        vopts.set("profile", "high");
    }
    let mut venc = venc.open_with(vopts).context("opening the H.264 encoder")?;

    // Audio encoder: AAC, stereo, 48 kHz.
    let acodec = ff::encoder::find(ff::codec::Id::AAC).context("this FFmpeg has no AAC encoder")?;
    let mut actx = ff::codec::context::Context::new_with_codec(acodec);
    actx.set_time_base(ff::Rational::new(1, AUDIO_RATE as i32));
    if global {
        actx.set_flags(ff::codec::Flags::GLOBAL_HEADER);
    }
    let mut aenc = actx.encoder().audio()?;
    aenc.set_rate(AUDIO_RATE as i32);
    aenc.set_channel_layout(ff::ChannelLayout::default(2));
    aenc.set_format(ff::format::Sample::F32(ff::format::sample::Type::Planar));
    aenc.set_bit_rate(AUDIO_BITRATE);
    let mut aenc = aenc.open_as(acodec).context("opening the AAC encoder")?;
    let afsize = (aenc.frame_size() as usize).max(1024);

    {
        let mut vs = octx.add_stream(vcodec)?;
        vs.set_parameters(&venc);
        vs.set_time_base(vtb);
        unsafe {
            (*vs.as_mut_ptr()).avg_frame_rate = ff::ffi::AVRational { num: fps.num as i32, den: fps.den as i32 };
        }
        let mut as_ = octx.add_stream(acodec)?;
        as_.set_parameters(&aenc);
        as_.set_time_base(ff::Rational::new(1, AUDIO_RATE as i32));
    }
    let mut hopts = ff::Dictionary::new();
    hopts.set("movflags", "+faststart");
    octx.write_header_with(hopts).context("writing the MP4 header")?;
    let vst = octx.stream(0).expect("added").time_base();
    let ast = octx.stream(1).expect("added").time_base();

    let mut sources: HashMap<usize, SourceRef> = HashMap::new();
    let mut open = |m: usize| -> Option<SourceRef> {
        if let Some(s) = sources.get(&m) {
            return Some(s.clone());
        }
        let s = media::open(job.read.get(m)?).ok()?;
        sources.insert(m, s.clone());
        Some(s)
    };
    let mut canvas = Canvas::new(w, h);
    let mut canvas_b = Canvas::new(w, h);
    let mut scaler = Scaler::default();
    // The viewer's colour passes on a device of our own (the CPU path is
    // only the fallback for a machine with no GPU).
    let mut gpu = crate::viewer::Offscreen::new(w, h).map(|o| (o, Idts::default()));
    let view = output_view(&job.seq);
    let mut texts: HashMap<u64, Option<crate::text::Bitmap>> = HashMap::new();
    let mut mixer = Mixer::new(plan_of(&job.seq, &|m| job.read.get(m).cloned()), AUDIO_RATE);
    let mut audio_buf: Vec<f32> = Vec::new();
    let mut audio_pts: i64 = 0;
    let (a, b) = job.range;
    let samples_at = |i: i64| -> i64 { (i as i128 * AUDIO_RATE as i128 * fps.den as i128 / fps.num as i128) as i64 };

    let drain_v = |enc: &mut ff::encoder::video::Encoder, octx: &mut ff::format::context::Output| -> Result<()> {
        let mut pkt = ff::Packet::empty();
        while enc.receive_packet(&mut pkt).is_ok() {
            pkt.set_stream(0);
            pkt.rescale_ts(vtb, vst);
            pkt.write_interleaved(octx)?;
        }
        Ok(())
    };
    let drain_a = |enc: &mut ff::encoder::audio::Encoder, octx: &mut ff::format::context::Output| -> Result<()> {
        let mut pkt = ff::Packet::empty();
        while enc.receive_packet(&mut pkt).is_ok() {
            pkt.set_stream(1);
            pkt.rescale_ts(ff::Rational::new(1, AUDIO_RATE as i32), ast);
            pkt.write_interleaved(octx)?;
        }
        Ok(())
    };

    for (i, t) in (a..b).enumerate() {
        if progress.cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        // Picture.
        canvas.clear();
        if let Some((off, idts)) = &mut gpu {
            let px = gpu_picture(off, idts, &job.seq, t, raster, (w, h), job.fit, &view, &mut |m, f| open(m).and_then(|s| s.decode(f).ok()));
            canvas.fill_rgba(&px);
        } else { match job.seq.picture_at(t) {
            Picture::Black => {}
            Picture::One(l) => {
                if let Some(f) = open(l.media).and_then(|s| s.decode(l.frame as u64).ok()) {
                    scaler.draw(&f, &mut canvas, raster, job.fit, l.reframe)?;
                }
            }
            Picture::Mix(la, lb, m) => {
                if let Some(f) = open(la.media).and_then(|s| s.decode(la.frame as u64).ok()) {
                    scaler.draw(&f, &mut canvas, raster, job.fit, la.reframe)?;
                }
                canvas_b.clear();
                if let Some(f) = open(lb.media).and_then(|s| s.decode(lb.frame as u64).ok()) {
                    scaler.draw(&f, &mut canvas_b, raster, job.fit, lb.reframe)?;
                }
                canvas.blend(&canvas_b, m);
            }
        } }
        draw_text(&mut canvas, &job.seq, t, raster, job.fit, &mut texts);
        let mut vf = canvas.frame();
        vf.set_pts(Some(i as i64));
        venc.send_frame(&vf)?;
        drain_v(&mut venc, &mut octx)?;

        // Sound up to the end of this frame, in whole encoder frames.
        let want = samples_at(i as i64 + 1);
        let have = audio_pts + (audio_buf.len() / 2) as i64;
        if want > have {
            let n = (want - have) as usize;
            let mut chunk = vec![0f32; n * 2];
            mixer.fill((t as f64 * fps.den as f64 / fps.num as f64) + (have - samples_at(i as i64)) as f64 / AUDIO_RATE as f64, &mut chunk);
            audio_buf.extend_from_slice(&chunk);
        }
        while audio_buf.len() / 2 >= afsize {
            let frame = audio_frame(&audio_buf[..afsize * 2], afsize, audio_pts);
            audio_buf.drain(..afsize * 2);
            audio_pts += afsize as i64;
            aenc.send_frame(&frame)?;
            drain_a(&mut aenc, &mut octx)?;
        }
        progress.done.store(i as u64 + 1, Ordering::Relaxed);
    }
    // The tail of the sound, padded to a whole encoder frame.
    if !audio_buf.is_empty() {
        let n = audio_buf.len() / 2;
        audio_buf.resize(afsize * 2, 0.0);
        let frame = audio_frame(&audio_buf, afsize, audio_pts);
        let _ = n;
        aenc.send_frame(&frame)?;
        drain_a(&mut aenc, &mut octx)?;
    }
    venc.send_eof()?;
    drain_v(&mut venc, &mut octx)?;
    aenc.send_eof()?;
    drain_a(&mut aenc, &mut octx)?;
    octx.write_trailer()?;
    Ok(vname)
}

/// The text over the picture at `t`, drawn into `c` where the raster sits
/// in it (made at that size, so the output is as sharp as it can be).
/// `cache` keeps each block's bitmap across frames.
fn draw_text(c: &mut Canvas, seq: &Sequence, t: i64, raster: (u32, u32), fit: Fit, cache: &mut HashMap<u64, Option<crate::text::Bitmap>>) {
    let layers = seq.text_at(t);
    if layers.is_empty() {
        return;
    }
    let (_, rr) = place(raster.0, raster.1, raster, (c.w, c.h), fit, model::Reframe::default());
    let size = (rr.w.round().max(1.0) as u32, rr.h.round().max(1.0) as u32);
    for l in layers {
        let k = crate::text::key(&l.text, &l.style, size);
        if cache.len() > 256 {
            cache.clear();
        }
        let b = cache.entry(k).or_insert_with(|| crate::text::rasterize(&l.text, &l.style, size));
        if let Some(b) = b {
            let b = b.clone();
            c.overlay(&b, rr.x.round() as i32 + b.x, rr.y.round() as i32 + b.y, l.opacity);
        }
    }
}

/// One frame of `seq` at record frame `t`, composed exactly as `run`
/// composes it (fit, reframes, dissolves, speed), as an RGB PNG whose
/// longest side is at most `max` (the raster's shape). `read[m]` is the file
/// media `m` is read from. What the scripting API's `viewer_capture` shows.
pub fn still(seq: &Sequence, read: &[PathBuf], t: i64, max: u32) -> Result<(Vec<u8>, u32, u32)> {
    media::movie::init();
    let raster = seq.raster.unwrap_or((1920, 1080));
    let s = (max.max(16) as f64 / raster.0.max(raster.1).max(1) as f64).min(1.0);
    let even = |v: f64| ((v.round() as u32).max(2) + 1) & !1;
    let (w, h) = (even(raster.0 as f64 * s), even(raster.1 as f64 * s));
    let mut canvas = Canvas::new(w, h);
    let mut scaler = Scaler::default();
    let decode = |l: model::Layer| -> Option<Frame> { media::open(read.get(l.media)?).ok()?.decode(l.frame.max(0) as u64).ok() };
    if let Some(mut off) = crate::viewer::Offscreen::new(w, h) {
        let px = gpu_picture(&mut off, &mut Idts::default(), seq, t, raster, (w, h), Fit::Fit, &output_view(seq), &mut |m, f| media::open(read.get(m)?).ok()?.decode(f).ok());
        canvas.fill_rgba(&px);
    } else { match seq.picture_at(t) {
        Picture::Black => {}
        Picture::One(l) => {
            if let Some(f) = decode(l) {
                scaler.draw(&f, &mut canvas, raster, Fit::Fit, l.reframe)?;
            }
        }
        Picture::Mix(la, lb, m) => {
            if let Some(f) = decode(la) {
                scaler.draw(&f, &mut canvas, raster, Fit::Fit, la.reframe)?;
            }
            let mut b = Canvas::new(w, h);
            if let Some(f) = decode(lb) {
                let mut sb = Scaler::default();
                sb.draw(&f, &mut b, raster, Fit::Fit, lb.reframe)?;
            }
            canvas.blend(&b, m);
        }
    } }
    draw_text(&mut canvas, seq, t, raster, Fit::Fit, &mut HashMap::new());
    // BT.709 limited-range 4:2:0 -> 8-bit RGB.
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    let (cs, ys) = (canvas.stride(1), canvas.stride(0));
    for y in 0..h as usize {
        for x in 0..w as usize {
            let yy = (canvas.planes[0][y * ys + x] as f32 - 16.0) * (255.0 / 219.0);
            let cb = (canvas.planes[1][(y / 2) * cs + x / 2] as f32 - 128.0) * (255.0 / 224.0);
            let cr = (canvas.planes[2][(y / 2) * cs + x / 2] as f32 - 128.0) * (255.0 / 224.0);
            let o = (y * w as usize + x) * 3;
            rgb[o] = (yy + 1.5748 * cr).round().clamp(0.0, 255.0) as u8;
            rgb[o + 1] = (yy - 0.1873 * cb - 0.4681 * cr).round().clamp(0.0, 255.0) as u8;
            rgb[o + 2] = (yy + 1.8556 * cb).round().clamp(0.0, 255.0) as u8;
        }
    }
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header()?;
        wr.write_image_data(&rgb)?;
    }
    Ok((out, w, h))
}

/// Interleaved stereo -> a planar float AAC input frame.
fn audio_frame(inter: &[f32], n: usize, pts: i64) -> ff::frame::Audio {
    let mut f = ff::frame::Audio::new(ff::format::Sample::F32(ff::format::sample::Type::Planar), n, ff::ChannelLayout::default(2));
    f.set_rate(AUDIO_RATE);
    for ch in 0..2 {
        let plane = f.plane_mut::<f32>(ch);
        for (k, o) in plane.iter_mut().take(n).enumerate() {
            *o = inter[k * 2 + ch];
        }
    }
    f.set_pts(Some(pts));
    f
}

/// The output picture: YUV 4:2:0, video range (black = 16, 128, 128).
struct Canvas {
    w: u32,
    h: u32,
    planes: [Vec<u8>; 3],
}

impl Canvas {
    /// Blend a premultiplied RGBA bitmap over the frame at (`ox`, `oy`)
    /// (output pixels), at `opacity`: BT.709 limited range, chroma from each
    /// 2x2 block's mean.
    fn overlay(&mut self, b: &crate::text::Bitmap, ox: i32, oy: i32, opacity: f32) {
        let (w, h) = (self.w as i32, self.h as i32);
        let at = |x: i32, y: i32| -> Option<[f32; 4]> {
            let (bx, by) = (x - ox, y - oy);
            if bx < 0 || by < 0 || bx >= b.w as i32 || by >= b.h as i32 {
                return None;
            }
            let i = (by as usize * b.w as usize + bx as usize) * 4;
            let p = &b.rgba[i..i + 4];
            Some([p[0] as f32 / 255.0 * opacity, p[1] as f32 / 255.0 * opacity, p[2] as f32 / 255.0 * opacity, p[3] as f32 / 255.0 * opacity])
        };
        let (x0, y0) = (ox.max(0), oy.max(0));
        let (x1, y1) = ((ox + b.w as i32).min(w), (oy + b.h as i32).min(h));
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let ys = self.stride(0);
        for y in y0..y1 {
            for x in x0..x1 {
                if let Some([r, g, bl, a]) = at(x, y) {
                    if a > 0.0 {
                        let i = y as usize * ys + x as usize;
                        let luma = 16.0 * a + 219.0 * (0.2126 * r + 0.7152 * g + 0.0722 * bl);
                        self.planes[0][i] = (self.planes[0][i] as f32 * (1.0 - a) + luma).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
        let cs = self.stride(1);
        for cy in (y0 / 2)..(y1 + 1) / 2 {
            for cx in (x0 / 2)..(x1 + 1) / 2 {
                let mut acc = [0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    if let Some(p) = at(cx * 2 + dx, cy * 2 + dy) {
                        for k in 0..4 {
                            acc[k] += p[k] / 4.0;
                        }
                    }
                }
                let [r, g, bl, a] = acc;
                if a <= 0.0 {
                    continue;
                }
                let i = cy as usize * cs + cx as usize;
                let cb = 128.0 * a + 224.0 * (-0.1146 * r - 0.3854 * g + 0.5 * bl);
                let cr = 128.0 * a + 224.0 * (0.5 * r - 0.4542 * g - 0.0458 * bl);
                if i < self.planes[1].len() {
                    self.planes[1][i] = (self.planes[1][i] as f32 * (1.0 - a) + cb).round().clamp(0.0, 255.0) as u8;
                    self.planes[2][i] = (self.planes[2][i] as f32 * (1.0 - a) + cr).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    /// The GPU's display-encoded RGBA, as BT.709 limited-range 4:2:0 (the
    /// encoder's input format; the colour itself was decided on the GPU).
    fn fill_rgba(&mut self, px: &[u8]) {
        let (w, h) = (self.w as usize, self.h as usize);
        let cs = self.stride(1);
        if px.len() < w * h * 4 {
            return;
        }
        let at = |x: usize, y: usize| {
            let o = (y.min(h - 1) * w + x.min(w - 1)) * 4;
            [px[o] as f32 / 255.0, px[o + 1] as f32 / 255.0, px[o + 2] as f32 / 255.0]
        };
        for y in 0..h {
            for x in 0..w {
                let [r, g, b] = at(x, y);
                self.planes[0][y * w + x] = (16.0 + 219.0 * (0.2126 * r + 0.7152 * g + 0.0722 * b)).round().clamp(0.0, 255.0) as u8;
            }
        }
        for cy in 0..h.div_ceil(2) {
            for cx in 0..w.div_ceil(2) {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let c = at(cx * 2 + dx, cy * 2 + dy);
                    for k in 0..3 {
                        acc[k] += c[k] * 0.25;
                    }
                }
                let [r, g, b] = acc;
                let yy = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                self.planes[1][cy * cs + cx] = (128.0 + 224.0 * (b - yy) / 1.8556).round().clamp(0.0, 255.0) as u8;
                self.planes[2][cy * cs + cx] = (128.0 + 224.0 * (r - yy) / 1.5748).round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    fn new(w: u32, h: u32) -> Self {
        let (cw, ch) = (w.div_ceil(2) as usize, h.div_ceil(2) as usize);
        Self { w, h, planes: [vec![16; (w * h) as usize], vec![128; cw * ch], vec![128; cw * ch]] }
    }

    fn clear(&mut self) {
        self.planes[0].fill(16);
        self.planes[1].fill(128);
        self.planes[2].fill(128);
    }

    fn stride(&self, p: usize) -> usize {
        if p == 0 { self.w as usize } else { self.w.div_ceil(2) as usize }
    }

    /// `self` = `self` × (1 − m) + `o` × m.
    fn blend(&mut self, o: &Canvas, m: f32) {
        let k = (m.clamp(0.0, 1.0) * 256.0) as u32;
        for p in 0..3 {
            for (a, b) in self.planes[p].iter_mut().zip(&o.planes[p]) {
                *a = ((*a as u32 * (256 - k) + *b as u32 * k) >> 8) as u8;
            }
        }
    }

    fn frame(&self) -> ff::frame::Video {
        let mut f = ff::frame::Video::new(ff::format::Pixel::YUV420P, self.w, self.h);
        for p in 0..3 {
            let (src_stride, dst_stride) = (self.stride(p), f.stride(p));
            let rows = if p == 0 { self.h as usize } else { self.h.div_ceil(2) as usize };
            let dst = f.data_mut(p);
            for y in 0..rows {
                dst[y * dst_stride..y * dst_stride + src_stride].copy_from_slice(&self.planes[p][y * src_stride..(y + 1) * src_stride]);
            }
        }
        unsafe {
            let fp = f.as_mut_ptr();
            (*fp).color_range = ff::ffi::AVColorRange::AVCOL_RANGE_MPEG;
            (*fp).colorspace = ff::ffi::AVColorSpace::AVCOL_SPC_BT709;
        }
        f
    }
}

/// swscale, cached across frames of the same shape.
#[derive(Default)]
struct Scaler {
    ctx: Option<(*mut ff::ffi::SwsContext, (i32, i32, i32, i32, i32))>,
    /// A half-float frame converted to 16-bit display code values.
    scratch: Vec<u16>,
}

impl Drop for Scaler {
    fn drop(&mut self) {
        if let Some((c, _)) = self.ctx.take() {
            unsafe { ff::ffi::sws_freeContext(c) };
        }
    }
}

/// A decoded frame as swscale input: format, plane pointers, strides.
struct Input {
    fmt: ff::ffi::AVPixelFormat,
    data: [*const u8; 4],
    stride: [i32; 4],
    w: u32,
    h: u32,
    matrix: Matrix,
    full: bool,
}

impl Scaler {
    fn input(&mut self, f: &Frame) -> Option<Input> {
        let (matrix, full) = match f.layout {
            Layout::Yuv { matrix, full_range, .. } => (matrix, full_range),
            _ => (Matrix::Bt709, true),
        };
        if let Some(av) = f.av_frame() {
            let fmt: ff::ffi::AVPixelFormat = av.format().into();
            let mut data = [std::ptr::null(); 4];
            let mut stride = [0i32; 4];
            for p in 0..av.planes().min(4) {
                data[p] = av.data(p).as_ptr();
                stride[p] = av.stride(p) as i32;
            }
            return Some(Input { fmt, data, stride, w: f.width, h: f.height, matrix, full });
        }
        let p = f.plane(0);
        let (fmt, data, stride) = match f.layout {
            Layout::Rgba8 => (ff::ffi::AVPixelFormat::AV_PIX_FMT_RGBA, p.data.as_ptr(), p.stride as i32),
            Layout::Rgba16 => (ff::ffi::AVPixelFormat::AV_PIX_FMT_RGBA64LE, p.data.as_ptr(), p.stride as i32),
            Layout::RgbaF16 => {
                // Scene-linear half floats -> sRGB-encoded 16-bit values.
                let n = (f.width * f.height * 4) as usize;
                self.scratch.resize(n, 0);
                for (i, o) in self.scratch.iter_mut().enumerate() {
                    let row = i / (f.width as usize * 4);
                    let col = i % (f.width as usize * 4);
                    let off = row * p.stride + col * 2;
                    let v = half::f16::from_bits(u16::from_le_bytes([p.data[off], p.data[off + 1]])).to_f32();
                    let v = if col % 4 == 3 { v.clamp(0.0, 1.0) } else { srgb(v) };
                    *o = (v * 65535.0 + 0.5) as u16;
                }
                (ff::ffi::AVPixelFormat::AV_PIX_FMT_RGBA64LE, self.scratch.as_ptr() as *const u8, (f.width * 8) as i32)
            }
            _ => return None,
        };
        Some(Input { fmt, data: [data, std::ptr::null(), std::ptr::null(), std::ptr::null()], stride: [stride, 0, 0, 0], w: f.width, h: f.height, matrix, full })
    }

    /// Scale the visible part of `f` into its place on `c`.
    fn draw(&mut self, f: &Frame, c: &mut Canvas, raster: (u32, u32), fit: Fit, reframe: model::Reframe) -> Result<()> {
        let Some(inp) = self.input(f) else { return Ok(()) };
        let (pl, rr) = place(inp.w, inp.h, raster, (c.w, c.h), fit, reframe);
        // The visible part (inside the picture, the raster and the output),
        // in output pixels (even, for 4:2:0), and the matching part of the
        // source.
        let x0 = pl.x.max(rr.x).max(0.0);
        let y0 = pl.y.max(rr.y).max(0.0);
        let x1 = (pl.x + pl.w).min(rr.x + rr.w).min(c.w as f64);
        let y1 = (pl.y + pl.h).min(rr.y + rr.h).min(c.h as f64);
        if x1 - x0 < 2.0 || y1 - y0 < 2.0 {
            return Ok(());
        }
        let (dx0, dy0) = ((x0.round() as i32) & !1, (y0.round() as i32) & !1);
        let (dx1, dy1) = ((x1.round() as i32) & !1, (y1.round() as i32) & !1);
        let (dw, dh) = (dx1 - dx0, dy1 - dy0);
        let sx = inp.w as f64 / pl.w;
        let sy = inp.h as f64 / pl.h;
        let desc = unsafe { ff::ffi::av_pix_fmt_desc_get(inp.fmt) };
        anyhow::ensure!(!desc.is_null(), "unknown pixel format");
        let (lw, lh) = unsafe { ((*desc).log2_chroma_w as i32, (*desc).log2_chroma_h as i32) };
        let align_x = 1 << lw.max(1);
        let align_y = 1 << lh.max(1);
        let cx0 = ((((dx0 as f64 - pl.x) * sx).max(0.0) as i32) / align_x) * align_x;
        let cy0 = ((((dy0 as f64 - pl.y) * sy).max(0.0) as i32) / align_y) * align_y;
        let cx1 = (((dx1 as f64 - pl.x) * sx).ceil() as i32).min(inp.w as i32);
        let cy1 = (((dy1 as f64 - pl.y) * sy).ceil() as i32).min(inp.h as i32);
        let (cw, ch) = (cx1 - cx0, cy1 - cy0);
        if cw < 2 || ch < 2 || dw < 2 || dh < 2 {
            return Ok(());
        }
        // Source plane pointers moved to the crop's corner.
        let mut src = inp.data;
        unsafe {
            for p in 0..4usize {
                if src[p].is_null() {
                    continue;
                }
                let comp = (0..(*desc).nb_components as usize).map(|k| (*desc).comp[k]).find(|c| c.plane as usize == p);
                let Some(comp) = comp else { continue };
                let chroma = p == 1 || p == 2;
                let (ox, oy) = if chroma { (cx0 >> lw, cy0 >> lh) } else { (cx0, cy0) };
                src[p] = src[p].offset(oy as isize * inp.stride[p] as isize + ox as isize * comp.step as isize);
            }
        }
        let key = (inp.fmt as i32, cw, ch, dw, dh);
        let ctx = match &self.ctx {
            Some((c, k)) if *k == key => *c,
            _ => {
                if let Some((c, _)) = self.ctx.take() {
                    unsafe { ff::ffi::sws_freeContext(c) };
                }
                let c = unsafe {
                    ff::ffi::sws_getContext(
                        cw,
                        ch,
                        inp.fmt,
                        dw,
                        dh,
                        ff::ffi::AVPixelFormat::AV_PIX_FMT_YUV420P,
                        ff::ffi::SwsFlags::SWS_BICUBIC as i32,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        std::ptr::null(),
                    )
                };
                anyhow::ensure!(!c.is_null(), "no scaler for this picture");
                self.ctx = Some((c, key));
                c
            }
        };
        let table = match inp.matrix {
            Matrix::Bt601 => ff::ffi::SWS_CS_ITU601,
            Matrix::Bt709 => ff::ffi::SWS_CS_ITU709,
            Matrix::Bt2020 => ff::ffi::SWS_CS_BT2020,
        };
        unsafe {
            let src_c = ff::ffi::sws_getCoefficients(table as i32);
            let dst_c = ff::ffi::sws_getCoefficients(ff::ffi::SWS_CS_ITU709 as i32);
            ff::ffi::sws_setColorspaceDetails(ctx, src_c, inp.full as i32, dst_c, 0, 0, 1 << 16, 1 << 16);
        }
        let mut dst: [*mut u8; 4] = [std::ptr::null_mut(); 4];
        let mut dst_stride = [0i32; 4];
        for p in 0..3 {
            let stride = c.stride(p);
            let (ox, oy) = if p == 0 { (dx0, dy0) } else { (dx0 / 2, dy0 / 2) };
            dst[p] = unsafe { c.planes[p].as_mut_ptr().add(oy as usize * stride + ox as usize) };
            dst_stride[p] = stride as i32;
        }
        unsafe {
            ff::ffi::sws_scale(ctx, src.as_ptr(), inp.stride.as_ptr(), 0, ch, dst.as_ptr(), dst_stride.as_ptr());
        }
        Ok(())
    }
}

fn srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_letterboxes_and_fill_crops() {
        // A 16:9 picture in a 16:9 raster, out to 9:16.
        let d = model::Reframe::default();
        let fit = place(1920, 1080, (1920, 1080), (1080, 1920), Fit::Fit, d).0;
        assert!((fit.w - 1080.0).abs() < 1e-6 && (fit.h - 607.5).abs() < 1e-6);
        assert!((fit.y - (1920.0 - 607.5) / 2.0).abs() < 1e-6);
        let fill = place(1920, 1080, (1920, 1080), (1080, 1920), Fit::Fill, d).0;
        assert!((fill.h - 1920.0).abs() < 1e-6, "fills the height");
        assert!(fill.x < 0.0, "and crops the sides");
        // A 4:3 picture in a 16:9 raster keeps its pillarbox.
        let p = place(1440, 1080, (1920, 1080), (1920, 1080), Fit::Fit, d).0;
        assert!((p.x - 240.0).abs() < 1e-6 && (p.w - 1440.0).abs() < 1e-6);
        // A 16:9 clip filled into a 9:16 raster and moved right: the output
        // shows a different part of it, and the raster bounds it.
        let r = model::Reframe { fill: true, x: 0.2, ..d };
        let (pic, raster) = place(1920, 1080, (1080, 1920), (1080, 1920), Fit::Fit, r);
        assert!((pic.h - 1920.0).abs() < 1e-6 && (pic.x - (-1166.6666666 + 216.0)).abs() < 1e-3);
        assert_eq!((raster.w, raster.h), (1080.0, 1920.0));
    }

    #[test]
    fn the_preset_follows_the_raster_shape() {
        assert_eq!(preset_for((1080, 1920)).id, "reels");
        assert_eq!(preset_for((3840, 2160)).id, "youtube-1080");
        assert_eq!(preset_for((1080, 1350)).id, "feed-4x5");
        assert_eq!(preset_for((1000, 1000)).id, "square");
    }

    #[test]
    fn a_taken_name_gets_a_number() {
        let dir = std::env::temp_dir().join(format!("scrub-free-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.mp4");
        assert_eq!(free_name(&p), p);
        std::fs::write(&p, b"x").unwrap();
        assert_eq!(free_name(&p), dir.join("a_2.mp4"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
