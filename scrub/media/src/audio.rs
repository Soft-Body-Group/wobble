//! A movie's sound: decoded through FFmpeg and resampled to interleaved
//! stereo f32 at the output device's rate, read forward from any sample.
//!
//! Positions are in OUTPUT samples (per channel) from the start of the
//! stream, so the timeline's arithmetic never sees the file's own rate.
//! A seek lands on the packet at or before the target and the samples
//! before it are dropped, so `read` after `seek(n)` starts exactly at n.

use anyhow::{Context as _, Result};
use ffmpeg_next as ff;
use std::collections::VecDeque;
use std::path::Path;

pub const CHANNELS: usize = 2;

pub struct AudioReader {
    ictx: ff::format::context::Input,
    stream: usize,
    decoder: ff::codec::decoder::Audio,
    resampler: Option<ff::software::resampling::Context>,
    rate: u32,
    tb: (i64, i64),
    start: i64,
    /// Decoded, resampled samples waiting to be read (interleaved).
    pending: VecDeque<f32>,
    /// Output sample position of `pending`'s front; None until located.
    pos: Option<i64>,
    /// After a seek: drop samples until this position.
    drop_until: i64,
    eof: bool,
}

impl AudioReader {
    /// `None` when the file has no audio.
    pub fn open(path: &Path, rate: u32) -> Result<Option<Self>> {
        crate::movie::init();
        let ictx = ff::format::input(path).with_context(|| format!("opening {}", path.display()))?;
        let Some(stream) = ictx.streams().best(ff::media::Type::Audio) else { return Ok(None) };
        let index = stream.index();
        let tb = stream.time_base();
        let start = match stream.start_time() {
            i64::MIN => 0,
            s => s,
        };
        let decoder = ff::codec::context::Context::from_parameters(stream.parameters())?.decoder().audio()?;
        Ok(Some(Self {
            ictx,
            stream: index,
            decoder,
            resampler: None,
            rate,
            tb: (tb.numerator() as i64, tb.denominator().max(1) as i64),
            start,
            pending: VecDeque::new(),
            pos: Some(0),
            drop_until: 0,
            eof: false,
        }))
    }

    /// Where the next `read` starts (output samples).
    pub fn position(&self) -> Option<i64> {
        self.pos
    }

    pub fn seek(&mut self, sample: i64) {
        let sample = sample.max(0);
        if self.pos == Some(sample) {
            return;
        }
        // Output samples -> stream time base.
        let ts = self.start + (sample as i128 * self.tb.1 as i128 / (self.rate as i128 * self.tb.0 as i128).max(1)) as i64;
        unsafe {
            ff::ffi::avformat_seek_file(self.ictx.as_mut_ptr(), self.stream as i32, i64::MIN, ts, ts, 0);
        }
        self.decoder.flush();
        self.resampler = None;
        self.pending.clear();
        self.pos = None;
        self.drop_until = sample;
        self.eof = false;
    }

    /// Fill `out` (interleaved stereo); silence past the end. Returns the
    /// samples per channel written from the file.
    pub fn read(&mut self, out: &mut [f32]) -> usize {
        let want = out.len();
        while self.pending.len() < want && !self.eof {
            if self.decode_some().is_err() {
                self.eof = true;
            }
        }
        let n = want.min(self.pending.len());
        for (o, s) in out.iter_mut().zip(self.pending.drain(..n)) {
            *o = s;
        }
        for o in &mut out[n..] {
            *o = 0.0;
        }
        let frames = n / CHANNELS;
        if let Some(p) = &mut self.pos {
            *p += (want / CHANNELS) as i64;
        }
        frames
    }

    /// Decode one packet's worth into `pending`.
    fn decode_some(&mut self) -> Result<()> {
        let mut fed = false;
        for (s, packet) in self.ictx.packets() {
            if s.index() == self.stream {
                self.decoder.send_packet(&packet)?;
                fed = true;
                break;
            }
        }
        if !fed {
            self.decoder.send_eof().ok();
        }
        let mut frame = ff::frame::Audio::empty();
        let mut got = false;
        while self.decoder.receive_frame(&mut frame).is_ok() {
            got = true;
            self.take(&frame)?;
        }
        if !fed && !got {
            anyhow::bail!("end of stream");
        }
        Ok(())
    }

    fn take(&mut self, frame: &ff::frame::Audio) -> Result<()> {
        if self.resampler.is_none() {
            let layout = if frame.channel_layout().channels() > 0 {
                frame.channel_layout()
            } else {
                ff::ChannelLayout::default(frame.channels() as i32)
            };
            self.resampler = Some(ff::software::resampling::Context::get(
                frame.format(),
                layout,
                frame.rate(),
                ff::format::Sample::F32(ff::format::sample::Type::Packed),
                ff::ChannelLayout::default(CHANNELS as i32),
                self.rate,
            )?);
        }
        let n_out = (frame.samples() as u64 * self.rate as u64 / frame.rate().max(1) as u64) as usize + 256;
        let mut out = ff::frame::Audio::new(
            ff::format::Sample::F32(ff::format::sample::Type::Packed),
            n_out,
            ff::ChannelLayout::default(CHANNELS as i32),
        );
        out.set_rate(self.rate);
        self.resampler.as_mut().expect("made above").run(frame, &mut out)?;
        let samples = out.samples();
        let bytes = &out.data(0)[..samples * CHANNELS * 4];
        // Where this frame starts, once, after a seek.
        if self.pos.is_none() {
            let pts = frame.pts().or_else(|| frame.timestamp()).unwrap_or(self.start);
            let p = ((pts - self.start) as i128 * self.tb.0 as i128 * self.rate as i128 / self.tb.1 as i128) as i64;
            self.pos = Some(p);
        }
        let mut skip = 0usize;
        if let Some(p) = self.pos {
            let have = p + (self.pending.len() / CHANNELS) as i64;
            if have < self.drop_until {
                skip = ((self.drop_until - have) as usize).min(samples);
                if self.pending.is_empty() {
                    self.pos = Some(p + skip as i64);
                }
            }
        }
        for c in bytes[skip * CHANNELS * 4..].chunks_exact(4) {
            self.pending.push_back(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine written by the ffmpeg CLI reads back as a sine, and a seek
    /// lands on the same samples a straight read gets to.
    #[test]
    fn a_seek_lands_on_the_samples_a_straight_read_reaches() {
        let dir = std::env::temp_dir().join(format!("scrub-audio-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tone.mov");
        let ok = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=2:sample_rate=44100", "-c:a", "pcm_s16le"])
            .arg(&path)
            .status()
            .is_ok_and(|s| s.success());
        if !ok {
            eprintln!("skipped: no ffmpeg CLI");
            return;
        }
        let mut a = AudioReader::open(&path, 48000).unwrap().unwrap();
        let mut straight = vec![0f32; 48000 * CHANNELS];
        a.read(&mut straight);
        let peak = straight.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.05, "silence where a tone should be ({peak})");
        let mut b = AudioReader::open(&path, 48000).unwrap().unwrap();
        b.seek(30000);
        let mut seeked = vec![0f32; 1000 * CHANNELS];
        b.read(&mut seeked);
        let err = seeked.iter().zip(&straight[30000 * CHANNELS..]).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        assert!(err < 0.02, "a seek to 30000 differs from reading to it by {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
