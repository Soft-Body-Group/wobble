//! Time-stretch that keeps pitch: WSOLA (waveform-similarity overlap-add).
//!
//! Sound at speed `s` is cut into Hann-windowed grains `W` long, laid down
//! every `Hs` output samples (50% overlap, so the windows sum to one) and
//! taken from the source every `Hs * s` samples. Each grain's start is
//! nudged by up to `DELTA` so its waveform lines up with what naturally
//! followed the previous grain: no phase jumps, so no warble, and the pitch
//! is the source's whatever the speed.
//!
//! Streaming and deterministic: output sample `j` is source time `j * s`
//! from where the stretch started (to within a grain's alignment nudge), so
//! playback and a render of the same stretch give the same samples. The
//! alignment search runs on a mono sum, coarse then fine, a few million
//! multiply-adds a second of sound.

use std::collections::VecDeque;

const CH: usize = 2;

pub struct Stretch {
    speed: f64,
    /// Output hop; a grain is `2 * hs` long.
    hs: usize,
    /// How far a grain may move to line up.
    delta: usize,
    win: Vec<f32>,
    /// Source sound (interleaved stereo) and its mono sum for the search.
    input: Vec<f32>,
    mono: Vec<f32>,
    /// Index (relative to where the stretch started) of `input`'s first frame.
    base: i64,
    /// The next grain.
    k: u64,
    /// Where the previous grain was taken from.
    prev: Option<i64>,
    /// The overlap-add of the grains not yet finished.
    acc: Vec<f32>,
    ready: VecDeque<f32>,
    scratch: Vec<f32>,
}

impl Stretch {
    /// A stretch at `speed` (> 0; 2 plays twice as fast) for sound at `rate`.
    pub fn new(speed: f64, rate: u32) -> Self {
        // ~21 ms hops, ~43 ms grains: short enough to keep transients,
        // long enough to hold a voice's lowest pitch period.
        let hs = ((rate as usize * 1024) / 48000).clamp(128, 4096);
        let delta = hs / 2;
        let w = 2 * hs;
        let win = (0..w).map(|n| (0.5 - 0.5 * (std::f64::consts::TAU * n as f64 / w as f64).cos()) as f32).collect();
        Self {
            speed: speed.max(1e-3),
            hs,
            delta,
            win,
            input: Vec::new(),
            mono: Vec::new(),
            base: -(Self::lead_for(hs, delta) as i64),
            k: 0,
            prev: None,
            acc: vec![0.0; w * CH],
            ready: VecDeque::new(),
            scratch: Vec::new(),
        }
    }

    fn lead_for(hs: usize, delta: usize) -> usize {
        hs + delta
    }

    /// How many samples before the starting point the source must begin:
    /// the first read is from `start - lead()`.
    pub fn lead(&self) -> usize {
        Self::lead_for(self.hs, self.delta)
    }

    /// Fill `out` (interleaved stereo) with stretched sound, pulling the
    /// source forward through `src` (which fills its slice with the next
    /// source samples, interleaved stereo, in order).
    pub fn process(&mut self, out: &mut [f32], src: &mut dyn FnMut(&mut [f32])) {
        while self.ready.len() < out.len() {
            self.grain(src);
        }
        let n = out.len();
        for (o, s) in out.iter_mut().zip(self.ready.drain(..n)) {
            *o = s;
        }
    }

    fn grain(&mut self, src: &mut dyn FnMut(&mut [f32])) {
        let (hs, delta, w) = (self.hs as i64, self.delta as i64, 2 * self.hs);
        // Grain k's centre (output k*hs + hs, which is output sample k*hs
        // once the first half grain is dropped) sits on source k*hs*speed.
        let nom = (self.k as f64 * hs as f64 * self.speed).round() as i64 - hs;
        let q = self.prev.map(|p| p + hs);
        let lo = q.map_or(nom - delta, |q| q.min(nom - delta));
        let hi = (nom + delta + w as i64).max(q.map_or(i64::MIN, |q| q + hs));
        self.drop_before(lo);
        self.fill_to(hi, src);
        let p = match q {
            Some(q) => self.search(nom, q),
            None => nom,
        };
        let off = (p - self.base) as usize * CH;
        for (i, (a, x)) in self.acc.iter_mut().zip(&self.input[off..off + w * CH]).enumerate() {
            *a += x * self.win[i / CH];
        }
        // The first half of the first grain is its fade-in: not sound.
        if self.k > 0 {
            self.ready.extend(&self.acc[..self.hs * CH]);
        }
        self.acc.copy_within(self.hs * CH.., 0);
        let n = self.acc.len();
        self.acc[n - self.hs * CH..].fill(0.0);
        self.prev = Some(p);
        self.k += 1;
    }

    /// The grain start in `nom ± delta` whose first half best matches the
    /// sound at `q` (what followed the previous grain).
    fn search(&self, nom: i64, q: i64) -> i64 {
        let m = &self.mono;
        let t = (q - self.base) as usize;
        let hs = self.hs;
        let score = |p: i64, step: usize| -> f32 {
            let c = (p - self.base) as usize;
            let (mut xy, mut yy) = (0f32, 0f32);
            let mut i = 0;
            while i < hs {
                let y = m[c + i];
                xy += m[t + i] * y;
                yy += y * y;
                i += step;
            }
            xy / (yy + 1e-9).sqrt()
        };
        let d = self.delta as i64;
        let (mut best, mut best_s) = (nom, score(nom, 4));
        let mut p = nom - d;
        while p <= nom + d {
            let s = score(p, 4);
            if s > best_s {
                (best, best_s) = (p, s);
            }
            p += 4;
        }
        let coarse = best;
        best_s = score(coarse, 1);
        for p in (coarse - 3).max(nom - d)..=(coarse + 3).min(nom + d) {
            let s = score(p, 1);
            if s > best_s {
                (best, best_s) = (p, s);
            }
        }
        best
    }

    fn drop_before(&mut self, lo: i64) {
        let n = ((lo - self.base).max(0) as usize).min(self.mono.len());
        // Drain in lumps, not every grain.
        if n >= 4 * self.hs || n == self.mono.len() {
            self.input.drain(..n * CH);
            self.mono.drain(..n);
            self.base += n as i64;
        }
    }

    fn fill_to(&mut self, hi: i64, src: &mut dyn FnMut(&mut [f32])) {
        let end = self.base + self.mono.len() as i64;
        if hi <= end {
            return;
        }
        // Read a little past what is needed so reads come in lumps.
        let n = ((hi - end) as usize).max(self.hs);
        self.scratch.clear();
        self.scratch.resize(n * CH, 0.0);
        src(&mut self.scratch);
        self.input.extend_from_slice(&self.scratch);
        self.mono.extend(self.scratch.chunks_exact(CH).map(|f| f[0] + f[1]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48000;

    /// A source of stereo samples from `f(i)`, `i` relative to the start
    /// (the stretch's lead is negative `i`).
    fn source(f: impl Fn(i64) -> f32, lead: usize) -> impl FnMut(&mut [f32]) {
        let mut i = -(lead as i64);
        move |buf: &mut [f32]| {
            for fr in buf.chunks_exact_mut(2) {
                let v = f(i);
                fr[0] = v;
                fr[1] = v;
                i += 1;
            }
        }
    }

    fn sine(hz: f64) -> impl Fn(i64) -> f32 {
        move |i| if i < 0 { 0.0 } else { (std::f64::consts::TAU * hz * i as f64 / RATE as f64).sin() as f32 * 0.5 }
    }

    /// Rising zero crossings a second, from the left channel.
    fn pitch(x: &[f32]) -> f64 {
        let l: Vec<f32> = x.chunks_exact(2).map(|f| f[0]).collect();
        let ups = l.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        ups as f64 * RATE as f64 / l.len() as f64
    }

    #[test]
    fn the_pitch_stays_put_at_any_speed() {
        for speed in [0.5, 0.8, 1.25, 2.0, 3.0] {
            let mut s = Stretch::new(speed, RATE);
            let mut src = source(sine(440.0), s.lead());
            let mut out = vec![0f32; RATE as usize * 2];
            s.process(&mut out, &mut src);
            // Skip the first grain or two.
            let hz = pitch(&out[9600..]);
            assert!((hz - 440.0).abs() < 6.0, "{speed}x plays at {hz:.1} Hz, not 440");
        }
    }

    #[test]
    fn output_follows_source_time_times_speed() {
        // A tone whose loudness ramps with source time: at output sample j
        // the loudness must be the source's at j * speed.
        let ramp = |i: i64| if i < 0 { 0.0 } else { (std::f64::consts::TAU * 1000.0 * i as f64 / RATE as f64).sin() as f32 * (i as f32 / (4.0 * RATE as f32)) };
        for speed in [0.5, 2.0] {
            let mut s = Stretch::new(speed, RATE);
            let mut src = source(ramp, s.lead());
            let n = (RATE as f64 * 1.5 / speed.max(1.0)) as usize;
            let mut out = vec![0f32; n * 2];
            s.process(&mut out, &mut src);
            for j in [n / 3, n / 2, n * 9 / 10] {
                let win = &out[(j - 480) * 2..(j + 480) * 2];
                let rms = (win.iter().map(|v| v * v).sum::<f32>() / win.len() as f32).sqrt();
                let want = (j as f64 * speed / (4.0 * RATE as f64)) as f32 / 2f32.sqrt();
                assert!((rms - want).abs() < want * 0.08 + 0.002, "{speed}x at {j}: loudness {rms:.4}, want {want:.4}");
            }
        }
    }

    #[test]
    fn speed_one_is_the_source() {
        let mut s = Stretch::new(1.0, RATE);
        let f = sine(300.0);
        let mut src = source(&f, s.lead());
        let mut out = vec![0f32; 20000 * 2];
        s.process(&mut out, &mut src);
        let err = (2000..20000).map(|j| (out[j * 2] - f(j as i64)).abs()).fold(0f32, f32::max);
        assert!(err < 1e-3, "1x differs from the source by {err}");
    }

    #[test]
    fn chunked_calls_give_the_same_samples_as_one() {
        let f = sine(523.0);
        let mut a = Stretch::new(1.7, RATE);
        let mut src = source(&f, a.lead());
        let mut whole = vec![0f32; 30000 * 2];
        a.process(&mut whole, &mut src);
        let mut b = Stretch::new(1.7, RATE);
        let mut src = source(&f, b.lead());
        let mut parts = Vec::new();
        for n in [1, 777, 1024, 5000, 23198] {
            let mut buf = vec![0f32; n * 2];
            b.process(&mut buf, &mut src);
            parts.extend(buf);
        }
        assert_eq!(whole, parts);
    }
}
