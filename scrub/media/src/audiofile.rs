//! Sound files (WAV, MP3, AIFF, FLAC, M4A…) as clips: a `Source` with no
//! picture of its own. Its frames are made, not decoded: the whole file's
//! waveform with the frame's position marked, so the source viewer shows
//! where you are and a bin tile has a poster. The sound itself plays
//! through `audio::AudioReader` like any movie's.
//!
//! The waveform is a few hundred peak columns read off the file the first
//! time a picture is asked for (off the UI thread: a decode worker or the
//! bin's thumbnailer), so opening one — or a project full of them, whose
//! clips sit on audio tracks and never ask — costs a header read.
//!
//! Frames count at the rate it is opened with: the editor opens it at the
//! sequence's rate, so its marks fall on record frames.

use crate::frame::{Frame, Layout};
use crate::source::{Source, SourceInfo};
use anyhow::{Context as _, Result};
use ffmpeg_next as ff;
use std::path::Path;
use std::sync::OnceLock;

/// Extensions opened as sound files.
pub const AUDIO_EXTS: &[&str] = &["wav", "mp3", "aif", "aiff", "flac", "m4a", "aac", "ogg", "opus", "wma", "caf"];

pub fn is_audio(path: &Path) -> bool {
    path.extension().map(|e| e.to_string_lossy().to_lowercase()).is_some_and(|e| AUDIO_EXTS.contains(&e.as_str()))
}

/// The made picture's size.
const W: u32 = 960;
const H: u32 = 540;
/// Peak columns (one per pixel across).
const COLS: usize = W as usize;
/// The rate the peaks are read at: plenty for a picture, cheap to resample.
const PEAK_RATE: u32 = 8000;

pub struct AudioFile {
    info: SourceInfo,
    /// Per column, the loudest sample (0..1); `None` when it would not read.
    peaks: OnceLock<Option<Vec<f32>>>,
}

impl AudioFile {
    /// Open a file that has sound (and no picture Scrub uses), counted in
    /// frames at `fps_num / fps_den`.
    pub fn open(path: &Path, fps_num: i64, fps_den: i64) -> Result<Self> {
        crate::movie::init();
        let (fps_num, fps_den) = if fps_num > 0 && fps_den > 0 { (fps_num, fps_den) } else { (24, 1) };
        let ictx = ff::format::input(path).with_context(|| format!("opening {}", path.display()))?;
        let stream = ictx.streams().best(ff::media::Type::Audio).with_context(|| format!("{} has no sound", path.display()))?;
        let tb = stream.time_base();
        let secs = if stream.duration() > 0 {
            stream.duration() as f64 * tb.numerator() as f64 / tb.denominator().max(1) as f64
        } else {
            ictx.duration().max(0) as f64 / ff::ffi::AV_TIME_BASE as f64
        };
        let params = stream.parameters();
        let (codec, rate, channels) = unsafe {
            let p = params.as_ptr();
            (ff::codec::Id::from((*p).codec_id).name().to_string(), (*p).sample_rate, (*p).ch_layout.nb_channels)
        };
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let ch = match channels {
            1 => "mono".to_string(),
            2 => "stereo".to_string(),
            n => format!("{n} ch"),
        };
        let info = SourceInfo {
            path: path.to_path_buf(),
            label: format!("{ext} {codec} {:.1} kHz {ch}", rate as f64 / 1000.0),
            width: W,
            height: H,
            frames: (secs * fps_num as f64 / fps_den as f64).ceil().max(1.0) as u64,
            first: 0,
            fps_num,
            fps_den,
            has_alpha: false,
            has_audio: true,
            intra: true,
            audio_only: true,
        };
        Ok(Self { info, peaks: OnceLock::new() })
    }

    fn peaks(&self) -> Option<&[f32]> {
        self.peaks.get_or_init(|| read_peaks(&self.info.path, self.info.frames as f64 * self.info.fps_den as f64 / self.info.fps_num as f64)).as_deref()
    }
}

/// The file's loudest sample per column, read at `PEAK_RATE`.
fn read_peaks(path: &Path, secs: f64) -> Option<Vec<f32>> {
    let t0 = std::time::Instant::now();
    let mut r = crate::audio::AudioReader::open(path, PEAK_RATE).ok()??;
    let total = (secs * PEAK_RATE as f64).ceil().max(1.0) as usize;
    let per = total.div_ceil(COLS).max(1);
    let mut peaks = vec![0f32; COLS];
    let mut buf = vec![0f32; 4096 * crate::audio::CHANNELS];
    let mut at = 0usize;
    while at < total {
        let got = r.read(&mut buf);
        if got == 0 {
            break;
        }
        for (i, s) in buf[..got * crate::audio::CHANNELS].chunks_exact(crate::audio::CHANNELS).enumerate() {
            let c = ((at + i) / per).min(COLS - 1);
            let v = s.iter().fold(0f32, |m, x| m.max(x.abs()));
            if v > peaks[c] {
                peaks[c] = v;
            }
        }
        at += got;
    }
    log::debug!("{}: waveform in {:.0} ms", path.display(), t0.elapsed().as_secs_f64() * 1000.0);
    Some(peaks)
}

/// The waveform with frame `f` of `frames` marked: played part bright,
/// the rest dim, a white playhead line.
fn picture(peaks: Option<&[f32]>, f: u64, frames: u64) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let mut px = vec![0u8; w * h * 4];
    for p in px.chunks_exact_mut(4) {
        p.copy_from_slice(&[16, 18, 20, 255]);
    }
    let head = ((f as f64 + 0.5) / frames.max(1) as f64 * w as f64) as usize;
    let mid = h / 2;
    let amp = (h as f32) * 0.42;
    // The centre line.
    for x in 0..w {
        let o = (mid * w + x) * 4;
        px[o..o + 4].copy_from_slice(&[52, 58, 64, 255]);
    }
    if let Some(peaks) = peaks {
        for (x, &v) in peaks.iter().enumerate().take(w) {
            // A gentle curve so quiet passages still show.
            let half = (v.clamp(0.0, 1.0).powf(0.7) * amp).round() as usize;
            let col: [u8; 4] = if x < head { [96, 196, 170, 255] } else { [50, 104, 92, 255] };
            for y in mid.saturating_sub(half)..=(mid + half).min(h - 1) {
                let o = (y * w + x) * 4;
                px[o..o + 4].copy_from_slice(&col);
            }
        }
    }
    for y in 0..h {
        for x in head.saturating_sub(1)..=(head).min(w - 1) {
            let o = (y * w + x) * 4;
            px[o..o + 4].copy_from_slice(&[235, 235, 235, 255]);
        }
    }
    px
}

impl Source for AudioFile {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn decode(&self, frame: u64) -> Result<Frame> {
        let t0 = std::time::Instant::now();
        let px = picture(self.peaks(), frame.min(self.info.frames.saturating_sub(1)), self.info.frames);
        let mut f = Frame::owned(W, H, Layout::Rgba8, vec![px]);
        f.decode_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A WAV and an MP3 written by the ffmpeg CLI open as sound clips of
    /// the right length, and their pictures show a waveform.
    #[test]
    fn wav_and_mp3_open_as_sound_clips() {
        let dir = std::env::temp_dir().join(format!("scrub-audiofile-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["tone.wav", "tone.mp3"] {
            let path = dir.join(name);
            let ok = std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=2:sample_rate=44100"])
                .arg(&path)
                .status()
                .is_ok_and(|s| s.success());
            if !ok {
                eprintln!("skipped: no ffmpeg CLI (or no mp3 encoder)");
                continue;
            }
            assert!(is_audio(&path));
            let src = crate::source::open(&path).unwrap();
            let info = src.info().clone();
            assert!(info.audio_only && info.has_audio, "{name}: {info:?}");
            // 2 s at 24 fps (an MP3's encoder delay may add a frame).
            assert!((48..=50).contains(&info.frames), "{name}: {} frames", info.frames);
            let f = src.decode(10).unwrap();
            assert_eq!((f.width, f.height, f.layout), (W, H, Layout::Rgba8));
            // Some column above and below the centre line is lit.
            let p = f.plane(0);
            let lit = |y: usize| (0..W as usize).any(|x| p.data[(y * p.stride) + x * 4 + 1] > 90);
            assert!(lit(H as usize / 2 - 60) && lit(H as usize / 2 + 60), "{name}: no waveform drawn");
            let at25 = AudioFile::open(&path, 25, 1).unwrap();
            assert!((50..=52).contains(&at25.info().frames));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
