//! Movie containers — ProRes, H.264, and whatever else the linked FFmpeg
//! can decode — behind the `movie` feature.
//!
//! **This is the slow path, and it is marked as one.** Every other reader
//! in Wobble answers "give me frame N" by opening a file; a movie has to
//! answer it by decoding forward from the nearest keyframe, because that
//! is what the format is. Two things keep it usable:
//!
//! 1. **Sessions are kept open and warm.** A pool of decoder contexts is
//!    parked per file, each remembering the frame it would produce next.
//!    Sequential playback — the overwhelmingly common case — checks out a
//!    session that is already sitting on the wanted frame and costs one
//!    `receive_frame`. Only a scrub that lands outside every warm session
//!    pays for a seek.
//! 2. **The pool is sized for the player's concurrency**, so the several
//!    ticks the player keeps in flight decode in parallel instead of
//!    queueing behind one shared context.
//!
//! Colour: frames come out as 16-bit CODE VALUES (no curve, no matrix
//! beyond YCbCr->RGB), same contract as `still.rs`. The YCbCr matrix and
//! the studio/full range ARE applied, because those are carried in the
//! stream's own metadata and are not a guess — but the transfer function
//! is left alone for the reader leaf's `InputSpace` to handle, where the
//! user can see it. Rec.709 is what `guess_for_extension` prefills.
//!
//! Alpha rides through when the codec has it (ProRes 4444); everything
//! else decodes opaque.

use crate::decode::{ExrDecoder, ImageInfo};
use anyhow::{anyhow, Context as _, Result};
use ffmpeg_next as ff;
use half::f16;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};

/// Warm decoder contexts kept alive between decodes. Sized a little above
/// the player's in-flight depth so concurrent ticks do not serialise, and
/// bounded because each one holds a full decoder plus reference frames.
const MAX_SESSIONS: usize = 8;

/// How far forward a session will decode to reach a frame before giving up
/// and seeking instead. A GOP is rarely longer than this, and decoding a
/// few frames is cheaper than a seek plus its own catch-up; but chasing a
/// far-away frame linearly is how a scrub turns into a hang.
const MAX_CATCHUP: u64 = 48;

fn init() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if let Err(e) = ff::init() {
            log::error!("ffmpeg init failed: {e}");
        }
        // FFmpeg's own logging is noisy and duplicates ours; only real
        // errors are worth a line.
        ff::util::log::set_level(ff::util::log::Level::Error);
    });
}

/// What a file's video stream is, read once from the header.
#[derive(Clone, Copy, Debug)]
pub struct MovieInfo {
    pub width: u32,
    pub height: u32,
    pub frames: u64,
    /// Frames per second, as the exact rational the container declares.
    pub fps_num: i64,
    pub fps_den: i64,
    /// The stream's AVERAGE rate (frames over duration), 0.0 when the
    /// container does not declare one. A variable-rate clip (a macOS screen
    /// recording) declares 120 or 240 nominal and averages ~55: this is the
    /// rate it actually plays at. Display only — indexing uses `fps_num/den`.
    pub avg_fps: f64,
}

/// Open a file and find its video stream, returning both plus the index.
fn open(path: &Path) -> Result<(ff::format::context::Input, usize)> {
    init();
    let ictx = ff::format::input(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let index = ictx
        .streams()
        .best(ff::media::Type::Video)
        .map(|s| s.index())
        .with_context(|| format!("{} has no video stream", path.display()))?;
    Ok((ictx, index))
}

pub fn probe(path: &Path) -> Result<MovieInfo> {
    let (ictx, index) = open(path)?;
    let stream = ictx.stream(index).expect("index came from this context");
    let params = stream.parameters();
    let ctx = ff::codec::context::Context::from_parameters(params)
        .with_context(|| format!("codec parameters of {}", path.display()))?;
    let video = ctx
        .decoder()
        .video()
        .with_context(|| format!("opening the video decoder for {}", path.display()))?;

    let (mut num, mut den) = rate_of(&stream);
    let avg = stream.avg_frame_rate();
    let avg_fps = if avg.numerator() > 0 && avg.denominator() > 0 {
        avg.numerator() as f64 / avg.denominator() as f64
    } else {
        0.0
    };
    if num <= 0 || den <= 0 {
        // Last resort: one frame per unit of the stream's time base. Better
        // a wrong rate than a division by zero — the frame COUNT below is
        // what the timeline actually uses.
        num = 1;
        den = 1;
    }

    // nb_frames is authoritative when the container carries it (QuickTime
    // and MP4 both do). Otherwise derive it from the duration, which is
    // what a stream with no index leaves us.
    let mut frames = stream.frames().max(0) as u64;
    if frames == 0 {
        let tb = stream.time_base();
        let dur = stream.duration();
        if dur > 0 && tb.denominator() > 0 {
            let secs = dur as f64 * tb.numerator() as f64 / tb.denominator() as f64;
            frames = (secs * num as f64 / den as f64).round().max(1.0) as u64;
        }
    }

    Ok(MovieInfo {
        width: video.width(),
        height: video.height(),
        frames: frames.max(1),
        fps_num: num,
        fps_den: den,
        avg_fps,
    })
}

/// The outcome of one decode run: either the wanted picture, or proof
/// that the seek landed PAST it and has to be retried from further back.
enum Landed {
    Frame(ff::frame::Video),
    Overshot(u64),
}

impl From<ff::frame::Video> for Landed {
    fn from(f: ff::frame::Video) -> Self {
        Landed::Frame(f)
    }
}

/// SAFETY: the only non-`Send` member is swscale's `*mut SwsContext`,
/// which FFmpeg allows to be used from any thread so long as it is used
/// from one at a time. A session lives either inside `MovieDecoder::idle`
/// (behind a `Mutex`) or checked out by exactly one decode job, so it is
/// never touched concurrently. The same holds for the demuxer and decoder
/// contexts it owns.
unsafe impl Send for Session {}

/// The frame rate to index by, nominal first.
///
/// `avg_frame_rate` is DERIVED — FFmpeg computes it from the duration and
/// the frame count — so on a short or slightly-padded clip it comes back as
/// something like 576/23 instead of 24, and every frame past the first
/// rounding boundary then resolves one off. `r_frame_rate` is the rate the
/// container declares, exact for the constant-rate material this reads.
/// The average is kept only as a fallback for a stream that declares none.
fn rate_of(stream: &ff::format::stream::Stream) -> (i64, i64) {
    for r in [stream.rate(), stream.avg_frame_rate()] {
        let (n, d) = (r.numerator() as i64, r.denominator() as i64);
        if n > 0 && d > 0 {
            return (n, d);
        }
    }
    (1, 1)
}

/// One open, warm decoder. `next` is the frame index the next successful
/// `receive_frame` will yield — the whole reason a session is worth
/// keeping rather than reopening.
struct Session {
    path: PathBuf,
    ictx: ff::format::context::Input,
    stream: usize,
    decoder: ff::codec::decoder::Video,
    scaler: Option<ff::software::scaling::Context>,
    next: Option<u64>,
    /// Stream time base and the rate, cached so the pts<->frame maths does
    /// not re-read the stream on every frame.
    tb_num: i64,
    tb_den: i64,
    fps_num: i64,
    fps_den: i64,
    start: i64,
}

impl Session {
    fn open(path: &Path) -> Result<Self> {
        let (ictx, stream_index) = open(path)?;
        let stream = ictx.stream(stream_index).expect("index came from this context");
        let tb = stream.time_base();
        let (fps_num, fps_den) = rate_of(&stream);
        let start = match stream.start_time() {
            i64::MIN => 0,
            s => s,
        };
        let params = stream.parameters();
        let mut decoder = ff::codec::context::Context::from_parameters(params)
            .with_context(|| format!("codec parameters of {}", path.display()))?
            .decoder()
            .video()
            .with_context(|| format!("opening the video decoder for {}", path.display()))?;
        // Let libavcodec use its own threads inside one session. The player
        // already runs several sessions at once; this is what makes a
        // SINGLE session keep up on ProRes, where slice threading is most
        // of the decode speed.
        decoder.set_threading(ff::codec::threading::Config {
            kind: ff::codec::threading::Type::Frame,
            count: 0,
        });
        Ok(Self {
            path: path.to_path_buf(),
            ictx,
            stream: stream_index,
            decoder,
            scaler: None,
            next: None,
            tb_num: tb.numerator() as i64,
            tb_den: tb.denominator().max(1) as i64,
            fps_num,
            fps_den,
            start,
        })
    }

    /// Presentation timestamp, in stream time base units, of frame `n`.
    fn pts_of(&self, n: u64) -> i64 {
        // n / fps seconds, expressed in time-base ticks:
        //   n * fps_den/fps_num * tb_den/tb_num
        let num = n as i128 * self.fps_den as i128 * self.tb_den as i128;
        let den = (self.fps_num as i128 * self.tb_num as i128).max(1);
        self.start + (num / den) as i64
    }

    /// Which frame a presentation timestamp belongs to. Rounds to nearest,
    /// so a container whose timestamps drift by a tick still lands on the
    /// frame a user would name.
    fn frame_of(&self, pts: i64) -> u64 {
        let t = (pts - self.start).max(0) as i128;
        let num = t * self.tb_num as i128 * self.fps_num as i128;
        let den = (self.tb_den as i128 * self.fps_den as i128).max(1);
        ((num + den / 2) / den).max(0) as u64
    }

    fn seek_to(&mut self, frame: u64) -> Result<()> {
        let target = self.pts_of(frame);
        // `Input::seek` passes stream index -1, which means AV_TIME_BASE
        // microseconds; we have an exact pts in the STREAM's time base, and
        // converting it to microseconds and back is how a seek lands one
        // frame off. So call avformat directly with our own stream index.
        //
        // max_ts = target (not i64::MAX) is what makes this land on the
        // keyframe at or BEFORE the wanted frame — decoding forward from
        // there is the only way to reconstruct an inter frame.
        let ret = unsafe {
            ff::ffi::avformat_seek_file(
                self.ictx.as_mut_ptr(),
                self.stream as i32,
                i64::MIN,
                target,
                target,
                0,
            )
        };
        anyhow::ensure!(
            ret >= 0,
            "seeking {} to frame {frame} failed",
            self.path.display()
        );
        self.decoder.flush();
        self.next = None;
        Ok(())
    }

    /// Decode until the picture for `frame` is in hand.
    ///
    /// Landing EARLY is free — extra pictures are decoded and dropped.
    /// Landing LATE is fatal, and it happens: avformat seeks on decode
    /// timestamps, so in a stream with B-frames a keyframe whose
    /// presentation time is after the target can still have a decode time
    /// before it, and the seek picks it. There is no timestamp we can ask
    /// for that rules that out in general, so instead the overshoot is
    /// DETECTED — the first picture out of a seek carries its own pts —
    /// and the seek is retried from further back until the wanted frame is
    /// inside the decoded run.
    fn take(&mut self, frame: u64) -> Result<ff::frame::Video> {
        let mut backoff = 0u64;
        loop {
            let warm = match self.next {
                Some(n) => frame >= n && frame - n <= MAX_CATCHUP,
                None => false,
            };
            if !warm {
                self.seek_to(frame.saturating_sub(backoff))?;
            }
            match self.run(frame)? {
                Landed::Frame(f) => return Ok(f),
                Landed::Overshot(at) => {
                    // Already reading from the very start and still past
                    // the frame: the file's timestamps disagree with its
                    // own declared rate and there is nothing left to try.
                    anyhow::ensure!(
                        frame > backoff,
                        "{}: frame {frame} could not be reached — the first \
                         decodable picture is frame {at}",
                        self.path.display()
                    );
                    backoff = (backoff * 4).max(8);
                }
            }
        }
    }

    /// One attempt: decode forward from wherever the stream is positioned.
    fn run(&mut self, frame: u64) -> Result<Landed> {
        let mut decoded = ff::frame::Video::empty();
        let mut eof = false;
        loop {
            // Drain whatever the decoder already holds before feeding it.
            while self.decoder.receive_frame(&mut decoded).is_ok() {
                // Once we know where we are, COUNT. Pictures come out of
                // the decoder in presentation order, one per frame, so the
                // counter is exact — while pts->frame depends on a declared
                // rate a container is free to get wrong. pts is used only to
                // locate the picture a seek landed on, which is the one
                // moment there is nothing to count from.
                let at = match self.next {
                    Some(n) => n,
                    None => {
                        let pts =
                            decoded.pts().or_else(|| decoded.timestamp()).unwrap_or(0);
                        let at = self.frame_of(pts);
                        if at > frame {
                            return Ok(Landed::Overshot(at));
                        }
                        at
                    }
                };
                self.next = Some(at + 1);
                if at >= frame {
                    return Ok(decoded.into());
                }
            }
            if eof {
                // Ran out of stream before reaching the frame. A container
                // whose declared length overshoots its real one does this;
                // a file with no pictures at all is an error, not a black
                // frame.
                self.next = None;
                return Err(anyhow!(
                    "{}: frame {frame} is past the end of the stream",
                    self.path.display()
                ));
            }
            let mut fed = false;
            for (stream, packet) in self.ictx.packets() {
                if stream.index() != self.stream {
                    continue;
                }
                self.decoder
                    .send_packet(&packet)
                    .with_context(|| format!("decoding {}", self.path.display()))?;
                fed = true;
                break;
            }
            if !fed {
                self.decoder.send_eof().ok();
                eof = true;
            }
        }
    }

    /// Convert one decoded picture into 16-bit RGBA. Held on the session so
    /// the sws context — which is not cheap to build — is made once.
    fn to_rgba(&mut self, src: &ff::frame::Video) -> Result<ff::frame::Video> {
        let rebuild = match &self.scaler {
            Some(s) => {
                s.input().format != src.format()
                    || s.input().width != src.width()
                    || s.input().height != src.height()
            }
            None => true,
        };
        if rebuild {
            let mut sc = rgba_scaler(src.format(), src.width(), src.height())?;
            set_colour_details(&mut sc, &self.decoder);
            self.scaler = Some(sc);
        }
        let mut dst = ff::frame::Video::empty();
        self.scaler
            .as_mut()
            .expect("just built")
            .run(src, &mut dst)
            .context("converting the frame to RGBA")?;
        Ok(dst)
    }
}

/// The YCbCr -> RGBA64LE converter for one picture size. Same size in and
/// out, so the filter only decides how chroma is upsampled.
///
/// **Not BILINEAR, FAST_BILINEAR or AREA.** In FFmpeg 8.1 — the one a
/// release bundles — those three turn an 8-bit yuv420p H.264 into
/// posterised, wrapped colour on aarch64 (skin magenta, a green screen
/// lime); BICUBIC, LANCZOS, SPLINE and POINT are right, and FFmpeg 9 is
/// right with all of them. A dev build links brew's FFmpeg, so it never
/// showed. `the_converter_gets_yuv420p_right` pins it on whichever FFmpeg
/// the test links, and needs no encoder (the release build has none).
fn rgba_scaler(
    format: ff::format::Pixel,
    width: u32,
    height: u32,
) -> Result<ff::software::scaling::Context> {
    ff::software::scaling::Context::get(
        format,
        width,
        height,
        ff::format::Pixel::RGBA64LE,
        width,
        height,
        ff::software::scaling::Flags::BICUBIC,
    )
    .context("building the YCbCr->RGB converter")
}

/// Point swscale at the matrix and range the STREAM declares, rather than
/// leaving it on its BT.601 limited-range default — which would give every
/// HD clip a visible hue and contrast error. Unspecified metadata keeps
/// FFmpeg's own resolution-based guess.
fn set_colour_details(
    sc: &mut ff::software::scaling::Context,
    dec: &ff::codec::decoder::Video,
) {
    use ff::color::Range;
    use ff::color::Space;
    let table = match dec.color_space() {
        Space::BT709 => ff::ffi::SWS_CS_ITU709,
        Space::SMPTE170M | Space::BT470BG => ff::ffi::SWS_CS_ITU601,
        Space::SMPTE240M => ff::ffi::SWS_CS_SMPTE240M,
        Space::BT2020NCL | Space::BT2020CL => ff::ffi::SWS_CS_BT2020,
        // Unspecified: FFmpeg already guesses by resolution, and guessing
        // differently here would only add a second wrong answer.
        _ => return,
    };
    let src_full = matches!(dec.color_range(), Range::JPEG) as i32;
    unsafe {
        let coeffs = ff::ffi::sws_getCoefficients(table as i32);
        // Destination is RGB, which is always full range.
        ff::ffi::sws_setColorspaceDetails(
            sc.as_mut_ptr(),
            coeffs,
            src_full,
            coeffs,
            1,
            0,
            1 << 16,
            1 << 16,
        );
    }
}

/// A pool of warm sessions. `Mutex<Vec<..>>` rather than a per-thread map:
/// the player's decode jobs land on arbitrary rayon threads, so a session
/// has to be findable by the frame it is sitting on, not by who opened it.
#[derive(Default)]
pub struct MovieDecoder {
    idle: Mutex<Vec<Session>>,
}

impl MovieDecoder {
    /// The best idle session for this (path, frame): one already sitting on
    /// the frame beats one that would have to seek, and any session on the
    /// right file beats opening a new context.
    fn check_out(&self, path: &Path, frame: u64) -> Option<Session> {
        let mut idle = self.idle.lock().ok()?;
        let mut best: Option<(usize, u64)> = None;
        for (i, s) in idle.iter().enumerate() {
            if s.path != path {
                continue;
            }
            // Cost: how far this session would decode forward. A session
            // that would have to seek scores worst but is still preferred
            // over opening a whole new decoder.
            let cost = match s.next {
                Some(n) if frame >= n && frame - n <= MAX_CATCHUP => frame - n,
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
            // Drop the least useful: a session that has not decoded
            // anything yet, else the oldest.
            let victim = idle.iter().position(|s| s.next.is_none()).unwrap_or(0);
            idle.swap_remove(victim);
        }
        idle.push(session);
    }
}

impl ExrDecoder for MovieDecoder {
    fn read_header(&self, path: &Path) -> Result<ImageInfo> {
        let m = probe(path)?;
        Ok(ImageInfo { width: m.width, height: m.height })
    }

    fn frame_count(&self, path: &Path) -> Result<u64> {
        Ok(probe(path)?.frames)
    }

    fn decode_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
    ) -> Result<()> {
        self.decode_frame_into(path, info, row_stride, None, 0, dst)
    }

    fn decode_frame_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        _layer: Option<&str>,
        frame: u64,
        dst: &mut [f16],
    ) -> Result<()> {
        let w = info.width as usize;
        let h = info.height as usize;
        anyhow::ensure!(row_stride >= w * 4, "row stride smaller than row");
        anyhow::ensure!(dst.len() >= row_stride * h, "decode dst too small");

        let mut session = match self.check_out(path, frame) {
            Some(s) => s,
            None => Session::open(path)?,
        };
        // A session that FAILS is dropped rather than returned to the pool:
        // its decoder may be mid-error, and reopening is cheap next to
        // handing out a wrong picture. Every `?` below is that case.
        let decoded = session.take(frame)?;
        let picture = session.to_rgba(&decoded)?;
        // The session itself is healthy from here on, whatever the picture
        // turns out to be, so it goes back before anything else can return.
        let ok = picture.width() as usize == w && picture.height() as usize == h;
        self.check_in(session);
        anyhow::ensure!(
            ok,
            "{} frame {frame} is {}x{} but the sequence is {w}x{h}",
            path.display(),
            picture.width(),
            picture.height()
        );

        // RGBA64LE: four little-endian u16 per pixel, rows padded by
        // swscale's own stride, which is why this copies row by row.
        let src_stride = picture.stride(0);
        let bytes = picture.data(0);
        let lut = code_to_half();
        for y in 0..h {
            let row = &bytes[y * src_stride..y * src_stride + w * 8];
            let out = &mut dst[y * row_stride..y * row_stride + w * 4];
            for (o, v) in out.iter_mut().zip(row.chunks_exact(2)) {
                *o = lut[u16::from_le_bytes([v[0], v[1]]) as usize];
            }
        }
        Ok(())
    }
}

/// Every 16-bit code value as the half the reader hands on, `v / 65535`.
///
/// A table, not `f16::from_f32` per sample: on x86_64 (no `f16c` target
/// feature in a stock build) `half` checks the CPU at RUNTIME on every
/// call and the loop cannot vectorise, while on Apple silicon the same
/// line is one instruction. A 4K frame is 33M samples, so the conversion
/// alone cost Windows tens of ms a frame that the Mac never paid. 128 KB,
/// built once, identical results everywhere.
fn code_to_half() -> &'static [f16] {
    static LUT: std::sync::OnceLock<Vec<f16>> = std::sync::OnceLock::new();
    LUT.get_or_init(|| (0..=u16::MAX).map(|v| f16::from_f32(v as f32 / 65535.0)).collect())
}

/// The H.264-or-better encoders we try for [`encode_mp4`], best first. The
/// last is FFmpeg's own MPEG-4 Part 2, built into every FFmpeg that has
/// encoders at all (no external library, LGPL).
const MP4_ENCODERS: [&str; 4] = ["h264_videotoolbox", "libx264", "libopenh264", "mpeg4"];

/// Can this build write an mp4 (`encode_mp4`)? A release build's FFmpeg is
/// decoders-only.
pub fn can_encode_mp4() -> bool {
    init();
    MP4_ENCODERS.iter().any(|n| ff::encoder::find_by_name(n).is_some())
}

/// Write `frames` (tight 8-bit sRGB RGB, `w`×`h` each) to an mp4 at `fps`,
/// Rec.709 YCbCr, tagged so. What a GenAI clip step sends a video model:
/// a picture to look at, not a master, so 4:2:0 and a generous bit rate.
/// `w` and `h` must be even.
pub fn encode_mp4(path: &Path, frames: &[Vec<u8>], w: u32, h: u32, fps: f32) -> Result<()> {
    init();
    if frames.is_empty() || w % 2 != 0 || h % 2 != 0 {
        return Err(anyhow!("encode_mp4: {} frames at {w}x{h} (both sides must be even)", frames.len()));
    }
    let codec = MP4_ENCODERS
        .iter()
        .find_map(|n| ff::encoder::find_by_name(n))
        .ok_or_else(|| anyhow!("this build's FFmpeg has no video encoder (release builds are decode-only)"))?;
    let mut octx = ff::format::output(&path).with_context(|| format!("open {}", path.display()))?;
    let global_header = octx.format().flags().contains(ff::format::Flags::GLOBAL_HEADER);
    // Whole-number rates stay exact; 23.976 and friends go over 1001.
    let tb = if (fps - fps.round()).abs() < 1e-3 {
        ff::Rational::new(fps.round().max(1.0) as i32, 1).invert()
    } else {
        ff::Rational::new(1001, (fps * 1001.0).round() as i32)
    };
    let mut enc = ff::codec::context::Context::new_with_codec(codec).encoder().video()?;
    let yuv = ff::format::Pixel::YUV420P;
    enc.set_width(w);
    enc.set_height(h);
    enc.set_format(yuv);
    enc.set_time_base(tb);
    enc.set_frame_rate(Some(tb.invert()));
    // ~0.2 bits per pixel per frame: 720p24 at about 4.5 Mbit/s.
    enc.set_bit_rate(((w as f64 * h as f64 * fps as f64) * 0.2).max(1e6) as usize);
    enc.set_gop(fps.round().max(1.0) as u32);
    enc.set_colorspace(ff::color::Space::BT709);
    enc.set_color_range(ff::color::Range::MPEG);
    if global_header {
        enc.set_flags(ff::codec::Flags::GLOBAL_HEADER);
    }
    let mut enc = enc.open_as(codec).context("opening the video encoder")?;
    let mut stream = octx.add_stream(codec)?;
    stream.set_parameters(&enc);
    stream.set_time_base(tb);
    let index = stream.index();
    octx.write_header()?;
    // After write_header: the muxer picks its own timescale.
    let stream_tb = octx.stream(index).unwrap().time_base();
    let mut sc = ff::software::scaling::Context::get(
        ff::format::Pixel::RGB24,
        w,
        h,
        yuv,
        w,
        h,
        ff::software::scaling::Flags::BICUBIC,
    )?;
    unsafe {
        let coeffs = ff::ffi::sws_getCoefficients(ff::ffi::SWS_CS_ITU709 as i32);
        // Full-range RGB in, studio-range Rec.709 YCbCr out.
        ff::ffi::sws_setColorspaceDetails(sc.as_mut_ptr(), coeffs, 1, coeffs, 0, 0, 1 << 16, 1 << 16);
    }
    let drain = |enc: &mut ff::codec::encoder::Video, octx: &mut ff::format::context::Output| -> Result<()> {
        let mut packet = ff::Packet::empty();
        while enc.receive_packet(&mut packet).is_ok() {
            packet.set_stream(index);
            // One frame long, each: left at zero the LAST frame has no
            // length, and 48 frames at 24 read as 1.958 s, not 2 — which a
            // model with a two-second minimum refuses (Wan 2.7 did).
            if packet.duration() == 0 {
                packet.set_duration(1);
            }
            packet.rescale_ts(tb, stream_tb);
            packet.write_interleaved(octx)?;
        }
        Ok(())
    };
    let row = w as usize * 3;
    for (n, px) in frames.iter().enumerate() {
        if px.len() != row * h as usize {
            return Err(anyhow!("encode_mp4: frame {n} is {} bytes, not {}", px.len(), row * h as usize));
        }
        let mut rgb = ff::frame::Video::new(ff::format::Pixel::RGB24, w, h);
        let stride = rgb.stride(0);
        let data = rgb.data_mut(0);
        for y in 0..h as usize {
            data[y * stride..y * stride + row].copy_from_slice(&px[y * row..(y + 1) * row]);
        }
        // A FRESH output frame each time: the encoder may hold on to it.
        let mut out = ff::frame::Video::new(yuv, w, h);
        sc.run(&rgb, &mut out)?;
        out.set_pts(Some(n as i64));
        out.set_color_space(ff::color::Space::BT709);
        out.set_color_range(ff::color::Range::MPEG);
        enc.send_frame(&out)?;
        drain(&mut enc, &mut octx)?;
    }
    enc.send_eof()?;
    drain(&mut enc, &mut octx)?;
    octx.write_trailer()?;
    Ok(())
}

/// The H.264 encoders a movie Render tries, best first. Unlike
/// [`MP4_ENCODERS`] there is no MPEG-4 Part 2 fallback: a file labelled
/// H.264 must be H.264.
const H264_ENCODERS: [&str; 3] = ["h264_videotoolbox", "libx264", "libopenh264"];

/// The encoder a movie Render of `format` would use, or why there is none.
fn movie_encoder(format: crate::graph::WriteFormat) -> Result<ff::Codec> {
    init();
    if format.prores_profile().is_some() {
        return ff::encoder::find_by_name("prores_ks").ok_or_else(|| {
            anyhow!(
                "this build's FFmpeg cannot encode ProRes (no prores_ks encoder: \
                 release builds ship a decode-only FFmpeg)"
            )
        });
    }
    match format {
        crate::graph::WriteFormat::H264 => H264_ENCODERS
            .iter()
            .find_map(|n| ff::encoder::find_by_name(n))
            .ok_or_else(|| {
                anyhow!(
                    "this build's FFmpeg has no H.264 encoder (needs VideoToolbox, \
                     libx264 or OpenH264; release builds ship a decode-only FFmpeg)"
                )
            }),
        _ => Err(anyhow!("{} is not a movie format", format.label())),
    }
}

/// Can this build write `format`? `Err` says why not, in words a Render
/// panel can show.
pub fn can_encode(format: crate::graph::WriteFormat) -> std::result::Result<(), String> {
    movie_encoder(format).map(|_| ()).map_err(|e| e.to_string())
}

/// One movie file being written, frame by frame and in order: what a
/// Render with a movie `WriteFormat` feeds. Frames go in as 16-bit RGBA
/// code values (the output space already applied); swscale makes them
/// Rec.709 YCbCr, studio range, at the codec's subsampling and depth.
pub struct MovieWriter {
    octx: ff::format::context::Output,
    enc: ff::codec::encoder::Video,
    sc: ff::software::scaling::Context,
    tb: ff::Rational,
    stream_tb: ff::Rational,
    index: usize,
    yuv: ff::format::Pixel,
    w: u32,
    h: u32,
    n: i64,
}

impl MovieWriter {
    /// Open `path` (its extension picks the container) for a `w`×`h` movie
    /// at `fps`. `w` and `h` must suit the codec's subsampling
    /// (`WriteParams::movie_size`). `rec709` tags the stream's primaries
    /// and transfer as Rec.709; otherwise only the matrix is tagged, since
    /// a label that does not match the pixels is worse than none.
    pub fn create(
        path: &Path,
        w: u32,
        h: u32,
        fps: f32,
        format: crate::graph::WriteFormat,
        rec709: bool,
    ) -> Result<Self> {
        let codec = movie_encoder(format)?;
        let (sx, sy) = format.subsampled();
        anyhow::ensure!(
            w > 0 && h > 0 && !(sx && w % 2 != 0) && !(sy && h % 2 != 0),
            "{} cannot be {w}x{h}: its chroma needs even sides",
            format.label()
        );
        let yuv = match format {
            crate::graph::WriteFormat::H264 => ff::format::Pixel::YUV420P,
            crate::graph::WriteFormat::ProRes4444 | crate::graph::WriteFormat::ProRes4444Xq => {
                ff::format::Pixel::YUVA444P10LE
            }
            _ => ff::format::Pixel::YUV422P10LE,
        };
        let mut octx = ff::format::output(&path).with_context(|| format!("open {}", path.display()))?;
        let global_header = octx.format().flags().contains(ff::format::Flags::GLOBAL_HEADER);
        // Whole-number rates stay exact; 23.976 and friends go over 1001.
        let tb = if (fps - fps.round()).abs() < 1e-3 {
            ff::Rational::new(fps.round().max(1.0) as i32, 1).invert()
        } else {
            ff::Rational::new(1001, (fps * 1001.0).round() as i32)
        };
        let mut ctx = ff::codec::context::Context::new_with_codec(codec);
        ctx.set_threading(ff::codec::threading::Config::kind(ff::codec::threading::Type::Frame));
        let mut enc = ctx.encoder().video()?;
        enc.set_width(w);
        enc.set_height(h);
        enc.set_format(yuv);
        enc.set_time_base(tb);
        enc.set_frame_rate(Some(tb.invert()));
        enc.set_colorspace(ff::color::Space::BT709);
        enc.set_color_range(ff::color::Range::MPEG);
        if rec709 {
            enc.set_color_primaries(ff::color::Primaries::BT709);
            enc.set_color_transfer_characteristic(ff::color::TransferCharacteristic::BT709);
        }
        let mut opts = ff::Dictionary::new();
        if let Some(profile) = format.prores_profile() {
            opts.set("profile", &profile.to_string());
            // Apple's own vendor tag: some players are fussy without it.
            opts.set("vendor", "apl0");
        } else {
            // A review file: generous, ~0.25 bits per pixel per frame (2K at
            // 24 is about 13 Mbit/s) for the encoders that take a rate, and
            // a near-transparent constant quality for libx264.
            enc.set_bit_rate(((w as f64 * h as f64 * fps as f64) * 0.25).max(2e6) as usize);
            enc.set_gop((fps.round() as u32).max(1) * 2);
            opts.set("crf", "16");
            opts.set("preset", "medium");
        }
        if global_header {
            enc.set_flags(ff::codec::Flags::GLOBAL_HEADER);
        }
        let enc = enc
            .open_as_with(codec, opts)
            .with_context(|| format!("opening the {} encoder", codec.name()))?;
        let mut stream = octx.add_stream(codec)?;
        stream.set_parameters(&enc);
        stream.set_time_base(tb);
        let index = stream.index();
        octx.write_header().with_context(|| format!("writing {}", path.display()))?;
        // After write_header: the muxer picks its own timescale.
        let stream_tb = octx.stream(index).unwrap().time_base();
        let mut sc = ff::software::scaling::Context::get(
            ff::format::Pixel::RGBA64LE,
            w,
            h,
            yuv,
            w,
            h,
            ff::software::scaling::Flags::BICUBIC | ff::software::scaling::Flags::ACCURATE_RND,
        )?;
        unsafe {
            let coeffs = ff::ffi::sws_getCoefficients(ff::ffi::SWS_CS_ITU709 as i32);
            // Full-range RGB in, studio-range Rec.709 YCbCr out.
            ff::ffi::sws_setColorspaceDetails(sc.as_mut_ptr(), coeffs, 1, coeffs, 0, 0, 1 << 16, 1 << 16);
        }
        Ok(Self { octx, enc, sc, tb, stream_tb, index, yuv, w, h, n: 0 })
    }

    /// The next frame: tight RGBA, four `u16` per pixel, `w`×`h`.
    pub fn push(&mut self, rgba: &[u16]) -> Result<()> {
        let row = self.w as usize * 4;
        anyhow::ensure!(
            rgba.len() == row * self.h as usize,
            "movie frame {} is {} samples, not {}",
            self.n,
            rgba.len(),
            row * self.h as usize
        );
        let mut src = ff::frame::Video::new(ff::format::Pixel::RGBA64LE, self.w, self.h);
        let stride = src.stride(0);
        let data = src.data_mut(0);
        for (y, line) in rgba.chunks_exact(row).enumerate() {
            let out = &mut data[y * stride..y * stride + row * 2];
            for (o, v) in out.chunks_exact_mut(2).zip(line) {
                o.copy_from_slice(&v.to_le_bytes());
            }
        }
        // A FRESH output frame each time: the encoder may hold on to it.
        let mut out = ff::frame::Video::new(self.yuv, self.w, self.h);
        self.sc.run(&src, &mut out)?;
        out.set_pts(Some(self.n));
        out.set_color_space(ff::color::Space::BT709);
        out.set_color_range(ff::color::Range::MPEG);
        self.enc.send_frame(&out)?;
        self.n += 1;
        self.drain()
    }

    fn drain(&mut self) -> Result<()> {
        let mut packet = ff::Packet::empty();
        while self.enc.receive_packet(&mut packet).is_ok() {
            packet.set_stream(self.index);
            // One frame long, each, or the last frame has no length.
            if packet.duration() == 0 {
                packet.set_duration(1);
            }
            packet.rescale_ts(self.tb, self.stream_tb);
            packet.write_interleaved(&mut self.octx)?;
        }
        Ok(())
    }

    /// Flush the encoder and close the file. Frames written: `frames()`.
    pub fn finish(mut self) -> Result<()> {
        self.enc.send_eof()?;
        self.drain()?;
        self.octx.write_trailer()?;
        Ok(())
    }

    pub fn frames(&self) -> i64 {
        self.n
    }
}

#[cfg(test)]
mod tests {
    use super::{ff, rgba_scaler};

    /// The converter against the BT.601 limited-range formula (swscale's
    /// default for a stream that declares no matrix) on an 8-bit yuv420p
    /// picture: luma ramps across, and each 32-row band holds one chroma —
    /// neutral, skin, green screen, the extremes. FFmpeg 8.1's bilinear
    /// path wrapped exactly these (skin to magenta, green to lime) by up to
    /// most of the range; a correct converter is within rounding.
    #[test]
    fn the_converter_gets_yuv420p_right() {
        super::init();
        let chromas: [(u8, u8); 6] =
            [(128, 128), (110, 160), (90, 70), (16, 240), (240, 16), (200, 200)];
        let (w, band) = (256u32, 32u32);
        let h = band * chromas.len() as u32;
        let mut src = ff::frame::Video::new(ff::format::Pixel::YUV420P, w, h);
        let luma = |x: u32| (16 + x * 219 / (w - 1)) as u8;
        let stride = src.stride(0);
        for y in 0..h as usize {
            for x in 0..w as usize {
                src.data_mut(0)[y * stride + x] = luma(x as u32);
            }
        }
        for (plane, pick) in [(1usize, 0usize), (2, 1)] {
            let stride = src.stride(plane);
            for cy in 0..(h / 2) as usize {
                let (u, v) = chromas[cy * 2 / band as usize];
                let c = if pick == 0 { u } else { v };
                for cx in 0..(w / 2) as usize {
                    src.data_mut(plane)[cy * stride + cx] = c;
                }
            }
        }

        let mut dst = ff::frame::Video::empty();
        rgba_scaler(src.format(), w, h).unwrap().run(&src, &mut dst).unwrap();
        let (bytes, dstride) = (dst.data(0), dst.stride(0));
        let mut worst = (0.0f32, 0u32, 0u32);
        for (b, &(u, v)) in chromas.iter().enumerate() {
            // Rows clear of the band edges, where chroma is interpolated.
            for y in b as u32 * band + 4..(b as u32 + 1) * band - 4 {
                for x in 0..w {
                    let (yy, cb, cr) =
                        (luma(x) as f32 - 16.0, u as f32 - 128.0, v as f32 - 128.0);
                    let want = [
                        1.164 * yy + 1.596 * cr,
                        1.164 * yy - 0.392 * cb - 0.813 * cr,
                        1.164 * yy + 2.017 * cb,
                    ];
                    for (c, want) in want.iter().enumerate() {
                        let at = y as usize * dstride + x as usize * 8 + c * 2;
                        let got = u16::from_le_bytes([bytes[at], bytes[at + 1]]) as f32
                            / 65535.0
                            * 255.0;
                        let err = (got - want.clamp(0.0, 255.0)).abs();
                        if err > worst.0 {
                            worst = (err, x, y);
                        }
                    }
                }
            }
        }
        assert!(
            worst.0 < 3.0,
            "yuv420p -> RGBA is {:.1} code values off at ({}, {})",
            worst.0,
            worst.1,
            worst.2
        );
    }

    #[test]
    fn the_code_value_table_is_the_division() {
        let lut = super::code_to_half();
        assert_eq!(lut.len(), 65536);
        for v in [0u16, 1, 255, 256, 4096, 32767, 32768, 65534, 65535] {
            let want = half::f16::from_f32(v as f32 / 65535.0);
            assert_eq!(lut[v as usize].to_bits(), want.to_bits(), "code {v}");
        }
    }

    #[test]
    fn frame_and_pts_are_inverses_at_ntsc_rates() {
        // 23.976 in a 24000-tick time base: the case where naive integer
        // maths drifts a frame after a few seconds.
        let s = |fps_num: i64, fps_den: i64, tb_den: i64| {
            (fps_num, fps_den, tb_den)
        };
        let (fps_num, fps_den, tb_den) = s(24000, 1001, 24000);
        // Mirror Session::pts_of / frame_of without opening a file.
        let pts_of = |n: u64| -> i64 {
            let num = n as i128 * fps_den as i128 * tb_den as i128;
            let den = fps_num as i128;
            (num / den) as i64
        };
        let frame_of = |pts: i64| -> u64 {
            let num = pts as i128 * fps_num as i128;
            let den = tb_den as i128 * fps_den as i128;
            ((num + den / 2) / den) as u64
        };
        for n in [0u64, 1, 23, 24, 100, 1000, 86399] {
            assert_eq!(frame_of(pts_of(n)), n, "frame {n} did not round-trip");
        }
    }
}
