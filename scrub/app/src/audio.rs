//! Sound: every audible audio track of the sequence (A1-A3 and M, mixed
//! with their gains, mute and solo) or the source clip's own track, through
//! the default output device, and the clock playback follows.
//!
//! A `Plan` is what is playing: the audio clips laid on the timeline and
//! the frame playback started from. A feeder thread renders it ahead into
//! a ring (~0.3 s), reading each clip's file through `media::audio` at the
//! source position the timeline asks for. The device callback drains the
//! ring and COUNTS what it played: that count is the clock. While sound
//! plays, the playhead is placed from it, so picture follows sound and a
//! stall in the feed holds the picture rather than letting them drift.
//!
//! Only 1x forward plays sound. Shuttling, reverse and speed-changed clips
//! are silent for now (no resampling of the speed, no scrub snippets).

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use media::audio::{AudioReader, CHANNELS};
use model::Fps;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// How much the feeder keeps rendered ahead, in seconds.
const AHEAD_SECS: f64 = 0.3;
const CHUNK: usize = 1024;

/// One stretch of sound on the timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanClip {
    pub path: PathBuf,
    pub start: i64,
    pub len: i64,
    pub src_in: i64,
    pub src_fps: Fps,
    pub speed: f64,
    /// The track's linear gain.
    pub gain: f32,
    /// The track it is on (clips on different tracks read separately).
    pub lane: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub clips: Vec<PlanClip>,
    pub fps: Fps,
}

/// A sequence's sound as a mixer plan: the clips of every audible track
/// with that track's gain. `read` gives the file each media item is read
/// from (an optimised copy, say); `None` reads the original.
pub fn plan_of(seq: &model::Sequence, read: &dyn Fn(usize) -> Option<PathBuf>) -> Plan {
    let mut clips = Vec::new();
    for (t, gain) in seq.audible() {
        if gain <= 0.0 {
            continue;
        }
        for c in &seq.tracks[t].clips {
            clips.push(PlanClip {
                path: read(c.media).unwrap_or_else(|| seq.media[c.media].path.clone()),
                start: c.start,
                len: c.len,
                src_in: c.src_in,
                src_fps: seq.media[c.media].fps,
                // Against real time: a conformed clip at another rate is
                // retimed, and retimed sound is not played.
                speed: c.realtime_speed(seq.fps, seq.media[c.media].fps),
                gain,
                lane: t,
            });
        }
    }
    clips.sort_by_key(|c| c.start);
    Plan { clips, fps: seq.fps }
}

/// One device callback: drop the sound owed for time already counted,
/// play what the ring has into `out` (interleaved, `channels` wide), and
/// owe whatever it lacked. Returns the frames of time this callback is
/// (all of `out`, played or silent).
fn pull(ring: &mut VecDeque<f32>, out: &mut [f32], channels: usize, owed: &mut u64) -> usize {
    let frames = out.len() / channels;
    let skip = (*owed as usize).min(ring.len() / CHANNELS);
    ring.drain(..skip * CHANNELS);
    *owed -= skip as u64;
    let have = (ring.len() / CHANNELS).min(frames);
    for f in 0..have {
        let (l, r) = (ring.pop_front().unwrap_or(0.0), ring.pop_front().unwrap_or(0.0));
        let o = &mut out[f * channels..(f + 1) * channels];
        o[0] = l;
        if channels > 1 {
            o[1] = r;
        }
    }
    *owed += (frames - have) as u64;
    frames
}

struct Shared {
    ring: Mutex<VecDeque<f32>>,
    /// Frames (per channel) of time the device has asked for since
    /// `start`: the clock. Counted whether or not the feeder had the sound
    /// ready, so a starved feeder costs silence, never time (a clock that
    /// counted only the samples played ran slow under starvation, and the
    /// picture, which follows it, played in slow motion).
    played: AtomicU64,
    /// Frames of sound owed for time already counted: dropped from the
    /// ring as they arrive, so the sound stays in step with the clock.
    owed: AtomicU64,
    active: AtomicBool,
    /// Bumped on every start/stop; a feeder from an older one quits.
    gen: AtomicU64,
}

pub struct Audio {
    _stream: cpal::Stream,
    shared: Arc<Shared>,
    pub rate: u32,
    feeder: Option<JoinHandle<()>>,
    /// What is playing and the frame it started from.
    playing: Option<(Plan, f64)>,
    /// The clock's last value and when it last moved (the stall watchdog).
    last_move: std::cell::Cell<(u64, std::time::Instant)>,
}

/// The clock may stand still this long while sound should be playing
/// before playback stops trusting it.
const STALL: std::time::Duration = std::time::Duration::from_millis(300);

impl Audio {
    /// The default output device, or why there is none.
    pub fn open() -> Result<Self, String> {
        let host = cpal::default_host();
        let dev = host.default_output_device().ok_or("no audio output device")?;
        let cfg = dev.default_output_config().map_err(|e| format!("audio config: {e}"))?;
        let rate = cfg.sample_rate();
        let channels = cfg.channels() as usize;
        let shared = Arc::new(Shared {
            ring: Mutex::new(VecDeque::new()),
            played: AtomicU64::new(0),
            owed: AtomicU64::new(0),
            active: AtomicBool::new(false),
            gen: AtomicU64::new(0),
        });
        let sh = shared.clone();
        // Headless checks run the clock without making a sound.
        let mute = media::sys::var_os("MUTE").is_some();
        let config = cpal::StreamConfig { channels: channels as u16, sample_rate: rate, buffer_size: cpal::BufferSize::Default };
        let stream = dev
            .build_output_stream::<f32, _, _>(
                config,
                move |out: &mut [f32], _| {
                    out.fill(0.0);
                    if !sh.active.load(Ordering::Acquire) {
                        return;
                    }
                    let frames = out.len() / channels;
                    // Never block the device thread on the feeder: a busy
                    // ring is silence now, and that much sound owed.
                    let Ok(mut ring) = sh.ring.try_lock() else {
                        sh.played.fetch_add(frames as u64, Ordering::Release);
                        sh.owed.fetch_add(frames as u64, Ordering::AcqRel);
                        return;
                    };
                    let mut owed = sh.owed.swap(0, Ordering::AcqRel);
                    let got = pull(&mut ring, out, channels, &mut owed);
                    sh.owed.fetch_add(owed, Ordering::AcqRel);
                    sh.played.fetch_add(got as u64, Ordering::Release);
                    if mute {
                        out.fill(0.0);
                    }
                },
                |e| log::warn!("audio stream: {e}"),
                None,
            )
            .map_err(|e| format!("audio stream: {e}"))?;
        stream.play().map_err(|e| format!("audio play: {e}"))?;
        Ok(Self { _stream: stream, shared, rate, feeder: None, playing: None, last_move: std::cell::Cell::new((0, std::time::Instant::now())) })
    }

    /// Frames buffered ahead of the device, and the clock in seconds.
    pub fn status(&self) -> (usize, f64) {
        let ring = self.shared.ring.lock().map(|r| r.len() / CHANNELS).unwrap_or(0);
        (ring, self.shared.played.load(Ordering::Acquire) as f64 / self.rate as f64)
    }

    pub fn is_playing(&self) -> bool {
        self.playing.is_some()
    }

    /// The plan playing, if any.
    pub fn plan(&self) -> Option<&Plan> {
        self.playing.as_ref().map(|(p, _)| p)
    }

    /// Play `plan` from record frame `from`.
    pub fn start(&mut self, plan: Plan, from: f64) {
        if media::sys::var_os("TRACE").is_some() {
            eprintln!("audio start at frame {from:.2}, {} clips", plan.clips.len());
        }
        self.stop();
        let gen = self.shared.gen.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.played.store(0, Ordering::Release);
        self.shared.owed.store(0, Ordering::Release);
        self.shared.ring.lock().unwrap().clear();
        let (sh, rate, p) = (self.shared.clone(), self.rate, plan.clone());
        self.feeder = std::thread::Builder::new().name("audio-feed".into()).spawn(move || feed(sh, gen, rate, p, from)).ok();
        // Let the feeder get ahead before the device starts counting.
        let t0 = std::time::Instant::now();
        while self.shared.ring.lock().unwrap().len() < CHUNK * CHANNELS * 2 && t0.elapsed().as_millis() < 100 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.shared.active.store(true, Ordering::Release);
        self.playing = Some((plan, from));
        self.last_move.set((0, std::time::Instant::now()));
    }

    pub fn stop(&mut self) {
        self.shared.active.store(false, Ordering::Release);
        self.shared.gen.fetch_add(1, Ordering::AcqRel);
        if let Some(f) = self.feeder.take() {
            let _ = f.join();
        }
        self.shared.ring.lock().unwrap().clear();
        self.playing = None;
    }

    /// The device has stopped taking samples (or never started): the clock
    /// has not moved for `STALL`. The caller then plays without sound.
    pub fn stalled(&self) -> bool {
        if self.playing.is_none() {
            return false;
        }
        let played = self.shared.played.load(Ordering::Acquire);
        let (last, at) = self.last_move.get();
        if played != last {
            self.last_move.set((played, std::time::Instant::now()));
            return false;
        }
        at.elapsed() > STALL
    }

    /// The record frame the sound has reached (the clock playback follows).
    pub fn clock(&self, fps: Fps) -> Option<f64> {
        let (_, from) = self.playing.as_ref()?;
        let secs = self.shared.played.load(Ordering::Acquire) as f64 / self.rate as f64;
        Some(from + secs * fps.as_f64())
    }
}

impl Drop for Audio {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Renders a plan's sound: interleaved stereo at `rate`, for any stretch of
/// timeline time. Playback's feeder and the export both use it, so a render
/// sounds exactly like playback.
pub struct Mixer {
    plan: Plan,
    rate: u32,
    /// One reader per (track, file): two tracks playing one file at once
    /// must not fight over a single read position.
    readers: HashMap<(usize, PathBuf), Option<AudioReader>>,
    scratch: Vec<f32>,
}

impl Mixer {
    pub fn new(plan: Plan, rate: u32) -> Self {
        Self { plan, rate, readers: HashMap::new(), scratch: Vec::new() }
    }

    /// Fill `out` (interleaved stereo) with the sound from `t0` seconds of
    /// timeline time on: every clip under that stretch, times its track's
    /// gain, summed. Speed-changed clips and gaps are silent.
    pub fn fill(&mut self, t0: f64, out: &mut [f32]) {
        out.fill(0.0);
        let frames = out.len() / CHANNELS;
        let (fps, rate) = (self.plan.fps.as_f64(), self.rate as f64);
        let t1 = t0 + frames as f64 / rate;
        // Clips are sorted by start: stop at the first that starts after.
        for k in 0..self.plan.clips.len() {
            let c = &self.plan.clips[k];
            let (cs, ce) = (c.start as f64 / fps, (c.start + c.len) as f64 / fps);
            if cs >= t1 {
                break;
            }
            if ce <= t0 || c.speed != 1.0 || c.gain <= 0.0 {
                continue;
            }
            // Output samples [a, b) of this buffer lie inside the clip.
            let a = (((cs - t0) * rate).ceil().max(0.0) as usize).min(frames);
            let b = (((ce - t0) * rate).ceil().max(0.0) as usize).min(frames);
            if a >= b {
                continue;
            }
            let t = t0 + a as f64 / rate;
            let src_secs = c.src_in as f64 / c.src_fps.as_f64() + (t - cs);
            let sample = (src_secs * rate).round() as i64;
            let (gain, key) = (c.gain, (c.lane, c.path.clone()));
            let reader = self.readers.entry(key.clone()).or_insert_with(|| AudioReader::open(&key.1, self.rate).ok().flatten());
            let Some(r) = reader else { continue };
            if r.position() != Some(sample) {
                r.seek(sample);
            }
            self.scratch.clear();
            self.scratch.resize((b - a) * CHANNELS, 0.0);
            r.read(&mut self.scratch);
            for (o, x) in out[a * CHANNELS..b * CHANNELS].iter_mut().zip(&self.scratch) {
                *o += x * gain;
            }
        }
    }
}

/// Render the plan from frame `from` into the ring until the generation moves on.
fn feed(sh: Arc<Shared>, gen: u64, rate: u32, plan: Plan, from: f64) {
    let from_secs = from / plan.fps.as_f64();
    let mut mixer = Mixer::new(plan, rate);
    let mut done: u64 = 0; // timeline samples rendered
    let mut buf = vec![0f32; CHUNK * CHANNELS];
    let cap = (AHEAD_SECS * rate as f64) as usize * CHANNELS;
    // (Tests: a slow feeder on purpose, ms a chunk.)
    let starve = media::sys::var("AUDIO_STARVE_MS").ok().and_then(|s| s.parse::<u64>().ok());
    while sh.gen.load(Ordering::Acquire) == gen {
        if sh.ring.lock().unwrap().len() >= cap {
            std::thread::sleep(std::time::Duration::from_millis(4));
            continue;
        }
        // Behind the clock with nothing queued: render from where the
        // clock is rather than sound that would only be skipped.
        if sh.ring.lock().unwrap().is_empty() {
            done += sh.owed.swap(0, Ordering::AcqRel);
        }
        if let Some(ms) = starve {
            std::thread::sleep(std::time::Duration::from_millis(ms));
        }
        mixer.fill(from_secs + done as f64 / rate as f64, &mut buf);
        done += CHUNK as u64;
        sh.ring.lock().unwrap().extend(buf.iter().copied());
    }
}

#[cfg(test)]
mod clock_tests {
    use super::*;

    #[test]
    fn a_starved_feeder_costs_silence_not_time() {
        // 1000 frames asked for, only 400 in the ring: the callback counts
        // 1000 (the clock keeps wall time) and owes 600, which are skipped
        // when they arrive.
        let mut ring: VecDeque<f32> = (0..400).flat_map(|i| [i as f32, i as f32]).collect();
        let mut out = vec![0.0f32; 1000 * 2];
        let mut owed = 0;
        assert_eq!(pull(&mut ring, &mut out, 2, &mut owed), 1000);
        assert_eq!(owed, 600);
        assert_eq!(out[2 * 399], 399.0);
        assert_eq!(out[2 * 400], 0.0);
        // The late sound arrives: the owed part is dropped, and playback
        // resumes where the clock is.
        ring.extend((400..2000).flat_map(|i| [i as f32, i as f32]));
        let mut out = vec![0.0f32; 10 * 2];
        pull(&mut ring, &mut out, 2, &mut owed);
        assert_eq!(owed, 0);
        assert_eq!(out[0], 1000.0);
    }
}
