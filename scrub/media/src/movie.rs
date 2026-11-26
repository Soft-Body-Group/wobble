//! Movie files through FFmpeg: ProRes, DNxHR, H.264, whatever the linked
//! FFmpeg decodes. Ported from Wobble's `engine/src/movie.rs` (its warm
//! session pool and seek rules), changed in three ways:
//!
//! 1. **Frames leave in the decoder's own pixel format, uncopied.** Wobble
//!    converts every picture to RGBA half on the CPU; here a planar YCbCr
//!    frame is handed on as the AVFrame itself (`Frame::av`) and the GPU
//!    converts it. Only an unusual format (NV12, packed RGB, a DPX's
//!    gbrp) goes through swscale, to 16-bit RGBA.
//! 2. **Pictures decoded on the way to a target are offered to the sink.**
//!    A long-GOP seek decodes every frame from the keyframe; reverse
//!    playback would otherwise decode the same GOP once per frame.
//! 3. **`intra` is read from the codec descriptor**, so the frame server
//!    can run ProRes frames in parallel and H.264 as one forward stream.
//!
//! Sessions are kept open and warm: a pool of decoder contexts parked per
//! file, each remembering the frame it would produce next. Sequential
//! playback checks out a session already sitting on the wanted frame and
//! costs one `receive_frame`; only a jump outside every warm session pays
//! for a seek.

use crate::frame::{Frame, Layout, Matrix};
use crate::source::{Sink, Source, SourceInfo};
use anyhow::{anyhow, Context as _, Result};
use ffmpeg_next as ff;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};
use std::time::Instant;

/// Warm decoder contexts kept alive between decodes. Bounded because each
/// holds a full decoder plus its reference frames.
const MAX_SESSIONS: usize = 8;

/// How far forward a session will decode to reach a frame before seeking
/// instead. A GOP is rarely longer; chasing a far frame linearly is how a
/// scrub turns into a hang.
pub const MAX_CATCHUP: u64 = 48;

pub fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if let Err(e) = ff::init() {
            log::error!("ffmpeg init failed: {e}");
        }
        ff::util::log::set_level(ff::util::log::Level::Error);
    });
}

fn open_input(path: &Path) -> Result<(ff::format::context::Input, usize)> {
    init();
    let ictx = ff::format::input(path).with_context(|| format!("opening {}", path.display()))?;
    let index = ictx
        .streams()
        .best(ff::media::Type::Video)
        .map(|s| s.index())
        .with_context(|| format!("{} has no video stream", path.display()))?;
    Ok((ictx, index))
}

/// The rate to index by: the container's nominal rate first. The average
/// is derived (576/23 on a short clip) and resolves frames one off.
fn rate_of(stream: &ff::format::stream::Stream) -> (i64, i64) {
    for r in [stream.rate(), stream.avg_frame_rate()] {
        let (n, d) = (r.numerator() as i64, r.denominator() as i64);
        if n > 0 && d > 0 {
            return (n, d);
        }
    }
    (24, 1)
}

/// Whether the codec's frames each decode alone (AV_CODEC_PROP_INTRA_ONLY).
fn codec_is_intra(id: ff::codec::Id) -> bool {
    unsafe {
        let d = ff::ffi::avcodec_descriptor_get(id.into());
        !d.is_null() && ((*d).props & ff::ffi::AV_CODEC_PROP_INTRA_ONLY as i32) != 0
    }
}

/// How each session threads. Frame threading is the throughput default;
/// slice threading has no pipeline to fill, so a seek costs less.
#[derive(Clone, Copy, Debug)]
pub struct Threads {
    pub slice: bool,
    pub count: usize,
    /// Decode on the media engine (VideoToolbox), macOS only.
    pub hw: bool,
}

impl Threads {
    /// `EDIFY_FF_THREADS=slice:4` / `frame:0` overrides (0 = FFmpeg picks).
    fn from_env(intra: bool, id: ff::codec::Id) -> Self {
        let hw = hw_wanted(intra, id);
        if let Ok(s) = crate::sys::var("FF_THREADS") {
            let (kind, n) = s.split_once(':').unwrap_or((&s, "0"));
            return Threads { slice: kind == "slice", count: n.parse().unwrap_or(0), hw };
        }
        // Frame threading holds back the first picture until `count`
        // packets are in, so EVERY seek decodes that many frames extra:
        // measured on 4K ProRes, a 700 ms seek against 50 with slices. An
        // intra codec seeks all the time (any worker, any frame) and gets
        // slice threads; a long-GOP stream mostly runs forward and gets
        // frame threads, the only kind x264 output can use.
        if intra {
            Threads { slice: true, count: 0, hw }
        } else {
            Threads { slice: false, count: 0, hw }
        }
    }
}

/// VideoToolbox for HEVC on macOS. Measured on an M4 Pro (4K, GOP 48),
/// through FFmpeg's hwaccel, which decodes one picture at a time:
/// - HEVC 10-bit 60 Mbit/s: seek 128 ms against 278 in software, scrub 64%
///   exact against 1%, playback 154 fps against 95. The media engine wins.
/// - H.264 50 Mbit/s: seek 222 ms against 108, playback 81 fps against 266.
///   Fourteen cores of libavcodec beat the media engine, so H.264 stays in
///   software.
/// `EDIFY_HW=0` turns it off, `=h264` adds H.264, `=all` tries every codec.
pub(crate) fn hw_wanted(intra: bool, id: ff::codec::Id) -> bool {
    if !cfg!(target_os = "macos") {
        return false;
    }
    match crate::sys::var("HW").as_deref() {
        Ok("0") => false,
        Ok("all") => true,
        Ok("h264") => !intra && matches!(id, ff::codec::Id::H264 | ff::codec::Id::HEVC),
        _ => !intra && id == ff::codec::Id::HEVC,
    }
}

/// libavcodec asks which format to decode into: VideoToolbox when offered.
unsafe extern "C" fn pick_videotoolbox(
    _ctx: *mut ff::ffi::AVCodecContext,
    fmts: *const ff::ffi::AVPixelFormat,
) -> ff::ffi::AVPixelFormat {
    let mut p = fmts;
    while *p != ff::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
        if *p == ff::ffi::AVPixelFormat::AV_PIX_FMT_VIDEOTOOLBOX {
            return *p;
        }
        p = p.add(1);
    }
    *fmts
}

/// Point a codec context at a fresh VideoToolbox device. False when the
/// machine has none (the caller then decodes in software).
pub(crate) fn attach_videotoolbox(ctx: &mut ff::codec::context::Context) -> bool {
    unsafe {
        let mut dev: *mut ff::ffi::AVBufferRef = std::ptr::null_mut();
        let r = ff::ffi::av_hwdevice_ctx_create(
            &mut dev,
            ff::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        );
        if r < 0 || dev.is_null() {
            return false;
        }
        let c = ctx.as_mut_ptr();
        (*c).hw_device_ctx = dev; // the context owns the reference now
        (*c).get_format = Some(pick_videotoolbox);
        true
    }
}

pub struct Movie {
    info: SourceInfo,
    threads: Threads,
    /// How far forward a warm session decodes rather than seek: 0 for an
    /// intra codec, where a seek is free and catching up decodes frames
    /// another worker is already decoding.
    catchup: u64,
    idle: Mutex<Vec<Session>>,
}

impl Movie {
    pub fn open(path: &Path) -> Result<Self> {
        let (ictx, index) = open_input(path)?;
        let stream = ictx.stream(index).expect("index came from this context");
        let params = stream.parameters();
        let id = params.id();
        let video = ff::codec::context::Context::from_parameters(params)
            .with_context(|| format!("codec parameters of {}", path.display()))?
            .decoder()
            .video()
            .with_context(|| format!("opening the video decoder for {}", path.display()))?;
        let (fps_num, fps_den) = rate_of(&stream);
        // nb_frames is authoritative when the container has it (QuickTime,
        // MP4); otherwise derive it from the duration.
        let mut frames = stream.frames().max(0) as u64;
        if frames == 0 {
            let tb = stream.time_base();
            let dur = if stream.duration() > 0 {
                stream.duration() as f64 * tb.numerator() as f64 / tb.denominator().max(1) as f64
            } else {
                ictx.duration().max(0) as f64 / ff::ffi::AV_TIME_BASE as f64
            };
            frames = (dur * fps_num as f64 / fps_den as f64).round().max(1.0) as u64;
        }
        let intra = codec_is_intra(id);
        let fmt = video.format();
        let has_alpha = matches!(native_layout(fmt, &video), Some(Layout::Yuv { alpha: true, .. }))
            || fmt.descriptor().is_some_and(|d| d.nb_components() == 4);
        let threads = Threads::from_env(intra, id);
        let name = if threads.hw { format!("{} videotoolbox", id.name()) } else { id.name().to_string() };
        let info = SourceInfo {
            path: path.to_path_buf(),
            label: format!("{name} {:?}", fmt).to_lowercase(),
            width: video.width(),
            height: video.height(),
            frames: frames.max(1),
            first: 0,
            fps_num,
            fps_den,
            has_alpha,
            has_audio: ictx.streams().best(ff::media::Type::Audio).is_some(),
            intra,
            audio_only: false,
        };
        Ok(Self { threads, catchup: if intra { 0 } else { MAX_CATCHUP }, info, idle: Mutex::new(Vec::new()) })
    }

    /// The best idle session for this frame: one sitting on it (or just
    /// before it) beats one that must seek, which beats opening one.
    fn check_out(&self, frame: u64) -> Option<Session> {
        let mut idle = self.idle.lock().ok()?;
        let mut best: Option<(usize, u64)> = None;
        for (i, s) in idle.iter().enumerate() {
            let cost = match s.next {
                Some(n) if frame >= n && frame - n <= self.catchup => frame - n,
                _ => u64::MAX - 1,
            };
            if best.is_none_or(|(_, c)| cost < c) {
                best = Some((i, cost));
            }
        }
        best.map(|(i, _)| idle.swap_remove(i))
    }

    fn check_in(&self, session: Session) {
        let Ok(mut idle) = self.idle.lock() else { return };
        if idle.len() >= MAX_SESSIONS {
            let victim = idle.iter().position(|s| s.next.is_none()).unwrap_or(0);
            idle.swap_remove(victim);
        }
        idle.push(session);
    }

    fn session(&self, frame: u64) -> Result<Session> {
        match self.check_out(frame) {
            Some(s) => Ok(s),
            None => Session::open(&self.info.path, self.threads, self.catchup),
        }
    }
}

impl Source for Movie {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn decode(&self, frame: u64) -> Result<Frame> {
        struct One(Option<Result<Frame>>);
        impl Sink for One {
            fn wants(&mut self, _: u64) -> bool {
                false
            }
            fn passing(&mut self, _: u64, _: Frame) {}
            fn done(&mut self, _: u64, p: Result<Frame>) -> Option<u64> {
                self.0 = Some(p);
                None
            }
        }
        let mut one = One(None);
        self.stream(frame, &mut one);
        one.0.unwrap_or_else(|| Err(anyhow!("no frame")))
    }

    fn stream(&self, first: u64, sink: &mut dyn Sink) {
        let mut session = match self.session(first) {
            Ok(s) => s,
            Err(e) => {
                sink.done(first, Err(e));
                return;
            }
        };
        let mut f = first;
        loop {
            let t0 = Instant::now();
            match session.take(f, sink) {
                Ok(mut picture) => {
                    picture.decode_ms = t0.elapsed().as_secs_f32() * 1000.0;
                    match sink.done(f, Ok(picture)) {
                        Some(n) => f = n,
                        None => break,
                    }
                }
                Err(e) => {
                    // A failed session may be mid-error: drop it, never
                    // pool it. The next target opens a fresh one.
                    let next = sink.done(f, Err(e));
                    match (next, Session::open(&self.info.path, self.threads, self.catchup)) {
                        (Some(n), Ok(s)) => {
                            session = s;
                            f = n;
                            continue;
                        }
                        _ => return,
                    }
                }
            }
        }
        self.check_in(session);
    }
}

/// The layout a decoder format can be handed on in, untouched.
fn native_layout(fmt: ff::format::Pixel, dec: &ff::codec::decoder::Video) -> Option<Layout> {
    use ff::format::Pixel as P;
    let (mut semi, mut shift) = (false, 0u8);
    let (bits, cx, cy, alpha, jpeg) = match fmt {
        P::NV12 => {
            semi = true;
            (8, 1, 1, false, false)
        }
        P::P010LE => {
            (semi, shift) = (true, 6);
            (10, 1, 1, false, false)
        }
        P::NV16 => {
            semi = true;
            (8, 1, 0, false, false)
        }
        P::P210LE => {
            (semi, shift) = (true, 6);
            (10, 1, 0, false, false)
        }
        P::YUV420P => (8, 1, 1, false, false),
        P::YUVJ420P => (8, 1, 1, false, true),
        P::YUV422P => (8, 1, 0, false, false),
        P::YUVJ422P => (8, 1, 0, false, true),
        P::YUV444P => (8, 0, 0, false, false),
        P::YUVJ444P => (8, 0, 0, false, true),
        P::YUVA420P => (8, 1, 1, true, false),
        P::YUVA444P => (8, 0, 0, true, false),
        P::YUV420P10LE => (10, 1, 1, false, false),
        P::YUV422P10LE => (10, 1, 0, false, false),
        P::YUV444P10LE => (10, 0, 0, false, false),
        P::YUVA422P10LE => (10, 1, 0, true, false),
        P::YUVA444P10LE => (10, 0, 0, true, false),
        P::YUV420P12LE => (12, 1, 1, false, false),
        P::YUV422P12LE => (12, 1, 0, false, false),
        P::YUV444P12LE => (12, 0, 0, false, false),
        P::YUVA444P12LE => (12, 0, 0, true, false),
        P::YUV420P16LE => (16, 1, 1, false, false),
        P::YUV422P16LE => (16, 1, 0, false, false),
        P::YUV444P16LE => (16, 0, 0, false, false),
        _ => return None,
    };
    use ff::color::{Range, Space};
    let matrix = match dec.color_space() {
        Space::BT709 => Matrix::Bt709,
        Space::SMPTE170M | Space::BT470BG => Matrix::Bt601,
        Space::BT2020NCL | Space::BT2020CL => Matrix::Bt2020,
        // Unspecified: SD is 601, everything else 709 (FFmpeg's own guess).
        _ if dec.height() < 720 => Matrix::Bt601,
        _ => Matrix::Bt709,
    };
    let full_range = jpeg || matches!(dec.color_range(), Range::JPEG);
    Some(Layout::Yuv { bits, cx, cy, alpha, matrix, full_range, semi, shift })
}

/// One open, warm decoder. `next` is the index the next `receive_frame`
/// yields: the reason a session is worth keeping.
struct Session {
    path: PathBuf,
    ictx: ff::format::context::Input,
    stream: usize,
    decoder: ff::codec::decoder::Video,
    scaler: Option<ff::software::scaling::Context>,
    next: Option<u64>,
    catchup: u64,
    tb_num: i64,
    tb_den: i64,
    fps_num: i64,
    fps_den: i64,
    start: i64,
}

/// SAFETY: the only non-Send member is swscale's context, which FFmpeg
/// allows on any thread one at a time. A session is either in the pool
/// (behind a Mutex) or checked out by exactly one decode job.
unsafe impl Send for Session {}

enum Landed {
    Frame(ff::frame::Video),
    Overshot(u64),
}

impl Session {
    fn open(path: &Path, threads: Threads, catchup: u64) -> Result<Self> {
        let (ictx, stream_index) = open_input(path)?;
        let stream = ictx.stream(stream_index).expect("index came from this context");
        let tb = stream.time_base();
        let (fps_num, fps_den) = rate_of(&stream);
        let start = match stream.start_time() {
            i64::MIN => 0,
            s => s,
        };
        let params = stream.parameters();
        let open_decoder = |hw: bool| -> Result<(ff::codec::decoder::Video, bool)> {
            let mut ctx = ff::codec::context::Context::from_parameters(params.clone())
                .with_context(|| format!("codec parameters of {}", path.display()))?;
            let hw = hw && attach_videotoolbox(&mut ctx);
            // Threading must be set before the decoder opens. The media
            // engine has no pipeline to fill; software frame threads do.
            ctx.set_threading(if hw {
                ff::codec::threading::Config { kind: ff::codec::threading::Type::Frame, count: 1 }
            } else {
                ff::codec::threading::Config {
                    kind: if threads.slice { ff::codec::threading::Type::Slice } else { ff::codec::threading::Type::Frame },
                    count: threads.count,
                }
            });
            let d = ctx.decoder().video().with_context(|| format!("opening the video decoder for {}", path.display()))?;
            Ok((d, hw))
        };
        let (decoder, _hw) = match open_decoder(threads.hw) {
            Ok(d) => d,
            Err(e) if threads.hw => {
                log::warn!("VideoToolbox would not open {}: {e:#}; decoding in software", path.display());
                open_decoder(false)?
            }
            Err(e) => return Err(e),
        };
        Ok(Self {
            path: path.to_path_buf(),
            ictx,
            stream: stream_index,
            decoder,
            scaler: None,
            next: None,
            catchup,
            tb_num: tb.numerator() as i64,
            tb_den: tb.denominator().max(1) as i64,
            fps_num,
            fps_den,
            start,
        })
    }

    fn pts_of(&self, n: u64) -> i64 {
        let num = n as i128 * self.fps_den as i128 * self.tb_den as i128;
        let den = (self.fps_num as i128 * self.tb_num as i128).max(1);
        self.start + (num / den) as i64
    }

    /// Nearest frame to a pts, so a tick of timestamp drift still lands on
    /// the frame a user would name.
    fn frame_of(&self, pts: i64) -> u64 {
        let t = (pts - self.start).max(0) as i128;
        let num = t * self.tb_num as i128 * self.fps_num as i128;
        let den = (self.tb_den as i128 * self.fps_den as i128).max(1);
        ((num + den / 2) / den).max(0) as u64
    }

    fn seek_to(&mut self, frame: u64) -> Result<()> {
        let target = self.pts_of(frame);
        // Our own stream index and an exact pts in its time base (not
        // Input::seek's microseconds); max_ts = target lands on the
        // keyframe at or BEFORE the frame.
        let ret = unsafe {
            ff::ffi::avformat_seek_file(self.ictx.as_mut_ptr(), self.stream as i32, i64::MIN, target, target, 0)
        };
        anyhow::ensure!(ret >= 0, "seeking {} to frame {frame} failed", self.path.display());
        self.decoder.flush();
        self.next = None;
        Ok(())
    }

    /// Decode until `frame` is in hand, as a `Frame`. A seek that lands
    /// PAST the frame (B-frames: avformat seeks on decode timestamps) is
    /// detected from the first picture's pts and retried from further back.
    fn take(&mut self, frame: u64, sink: &mut dyn Sink) -> Result<Frame> {
        let mut backoff = 0u64;
        let raw = loop {
            let warm = matches!(self.next, Some(n) if frame >= n && frame - n <= self.catchup);
            if !warm {
                self.seek_to(frame.saturating_sub(backoff))?;
            }
            match self.run(frame, sink)? {
                Landed::Frame(f) => break f,
                Landed::Overshot(at) => {
                    anyhow::ensure!(
                        frame > backoff,
                        "{}: frame {frame} could not be reached; the first decodable picture is {at}",
                        self.path.display()
                    );
                    backoff = (backoff * 4).max(8);
                }
            }
        };
        self.wrap(raw)
    }

    fn run(&mut self, frame: u64, sink: &mut dyn Sink) -> Result<Landed> {
        let mut eof = false;
        loop {
            loop {
                let mut decoded = ff::frame::Video::empty();
                if self.decoder.receive_frame(&mut decoded).is_err() {
                    break;
                }
                // Once located, COUNT: pictures leave in presentation
                // order, one per frame. pts only locates a seek's landing.
                let at = match self.next {
                    Some(n) => n,
                    None => {
                        let pts = decoded.pts().or_else(|| decoded.timestamp()).unwrap_or(0);
                        let at = self.frame_of(pts);
                        if at > frame {
                            return Ok(Landed::Overshot(at));
                        }
                        at
                    }
                };
                self.next = Some(at + 1);
                if at >= frame {
                    return Ok(Landed::Frame(decoded));
                }
                if sink.wants(at) {
                    if let Ok(p) = self.wrap(decoded) {
                        sink.passing(at, p);
                    }
                }
            }
            if eof {
                self.next = None;
                return Err(anyhow!("{}: frame {frame} is past the end of the stream", self.path.display()));
            }
            let mut fed = false;
            for (stream, packet) in self.ictx.packets() {
                if stream.index() != self.stream {
                    continue;
                }
                self.decoder.send_packet(&packet).with_context(|| format!("decoding {}", self.path.display()))?;
                fed = true;
                break;
            }
            if !fed {
                self.decoder.send_eof().ok();
                eof = true;
            }
        }
    }

    /// Hand a decoded picture on: as is when its format is one the GPU
    /// converts, else through swscale to 16-bit RGBA.
    fn wrap(&mut self, src: ff::frame::Video) -> Result<Frame> {
        let src = if src.format() == ff::format::Pixel::VIDEOTOOLBOX {
            // Down from the media engine's surface into system memory: one
            // copy (unified memory, so a memcpy), NV12 or P010.
            let mut sw = ff::frame::Video::empty();
            let r = unsafe { ff::ffi::av_hwframe_transfer_data(sw.as_mut_ptr(), src.as_ptr(), 0) };
            anyhow::ensure!(r >= 0, "reading back a VideoToolbox frame failed ({r})");
            sw
        } else {
            src
        };
        if let Some(layout) = native_layout(src.format(), &self.decoder) {
            return Ok(Frame::av(layout, src));
        }
        let rebuild = match &self.scaler {
            Some(s) => {
                s.input().format != src.format() || s.input().width != src.width() || s.input().height != src.height()
            }
            None => true,
        };
        if rebuild {
            // BICUBIC: FFmpeg 8.1's bilinear paths posterise 8-bit 4:2:0
            // on aarch64 (see Wobble's movie.rs).
            self.scaler = Some(
                ff::software::scaling::Context::get(
                    src.format(),
                    src.width(),
                    src.height(),
                    ff::format::Pixel::RGBA64LE,
                    src.width(),
                    src.height(),
                    ff::software::scaling::Flags::BICUBIC,
                )
                .context("building the RGBA converter")?,
            );
        }
        let mut dst = ff::frame::Video::empty();
        self.scaler.as_mut().expect("just built").run(&src, &mut dst).context("converting to RGBA")?;
        Ok(Frame::av(Layout::Rgba16, dst))
    }
}

/// Decode a single-picture file FFmpeg reads (DPX, and anything else with
/// an image2 demuxer) to a `Frame`.
pub fn decode_image(path: &Path) -> Result<Frame> {
    struct Last(Option<Result<Frame>>);
    impl Sink for Last {
        fn wants(&mut self, _: u64) -> bool {
            false
        }
        fn passing(&mut self, _: u64, _: Frame) {}
        fn done(&mut self, _: u64, p: Result<Frame>) -> Option<u64> {
            self.0 = Some(p);
            None
        }
    }
    let mut s = Session::open(path, Threads { slice: true, count: 0, hw: false }, 0)?;
    // One picture at the start: nothing to seek to.
    s.next = Some(0);
    s.take(0, &mut Last(None))
}

/// Size of a single-picture file FFmpeg reads.
pub fn image_size(path: &Path) -> Result<(u32, u32)> {
    let (ictx, index) = open_input(path)?;
    let stream = ictx.stream(index).expect("index came from this context");
    let video = ff::codec::context::Context::from_parameters(stream.parameters())?.decoder().video()?;
    Ok((video.width(), video.height()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn frame_and_pts_are_inverses_at_ntsc_rates() {
        let (fps_num, fps_den, tb_den) = (24000i64, 1001i64, 24000i64);
        let pts_of = |n: u64| (n as i128 * fps_den as i128 * tb_den as i128 / fps_num as i128) as i64;
        let frame_of = |pts: i64| {
            let num = pts as i128 * fps_num as i128;
            let den = tb_den as i128 * fps_den as i128;
            ((num + den / 2) / den) as u64
        };
        for n in [0u64, 1, 23, 24, 100, 1000, 86399] {
            assert_eq!(frame_of(pts_of(n)), n, "frame {n} did not round-trip");
        }
    }

    #[test]
    fn prores_is_intra_and_h264_is_not() {
        super::init();
        assert!(super::codec_is_intra(ffmpeg_next::codec::Id::PRORES));
        assert!(!super::codec_is_intra(ffmpeg_next::codec::Id::H264));
    }
}
