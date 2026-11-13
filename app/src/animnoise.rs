//! Noise ON a curve — Blender's F-curve Noise modifier. The keyed value
//! plus a wiggle, so a keyed move can shake (or a held value drift) without
//! keying the shake.
//!
//! It lives on the [`crate::anim::AnimCurve`] and is added inside
//! `eval_at`, so everything that evaluates a channel — playback, the curve
//! editor, motion paths, the properties rows — sees the same number. The
//! keys themselves never move: take the noise off and the curve is exactly
//! what was keyed.

use std::f32::consts::TAU;

/// The modifier's settings, named as Blender names them.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct CurveNoise {
    /// Frames per wiggle: bigger is slower.
    pub scale: f32,
    /// How far it swings, in the channel's own units: the wiggle stays
    /// within ± half of this.
    pub strength: f32,
    /// Where in the noise it reads — a seed, but a smooth one: animating
    /// it morphs the wiggle rather than cutting to another.
    pub phase: f32,
    /// Slides the wiggle along the timeline, in frames.
    pub offset: f32,
    /// Extra octaves of finer detail; 0 is one smooth layer.
    pub depth: u32,
}

impl Default for CurveNoise {
    fn default() -> Self {
        Self { scale: 10.0, strength: 1.0, phase: 1.0, offset: 0.0, depth: 0 }
    }
}

/// Octave cap for `depth`.
pub const MAX_DEPTH: u32 = 7;

impl CurveNoise {
    /// A noise sized to the curve it goes on: a tenth of the bigger of its
    /// value span and its magnitude, so a 500 px move shakes by tens of
    /// pixels and an opacity by a few percent. A curve sitting at 0 gets 1.
    pub fn sized_for(values: impl Iterator<Item = f32>) -> Self {
        let (mut lo, mut hi, mut mag) = (f32::MAX, f32::MIN, 0.0f32);
        for v in values {
            lo = lo.min(v);
            hi = hi.max(v);
            mag = mag.max(v.abs());
        }
        let m = if lo <= hi { (hi - lo).max(mag) } else { 0.0 };
        Self { strength: if m > 1e-6 { m * 0.1 } else { 1.0 }, ..Default::default() }
    }

    /// What it adds at frame `t`: centred on 0, within ± strength / 2.
    pub fn at(&self, t: f32) -> f32 {
        let x = (t - self.offset) / self.scale.abs().max(1e-3);
        // Off the lattice in y, or every whole `scale` frames would read 0
        // at an integer phase (gradient noise is 0 on its lattice points).
        let y = self.phase + 0.37;
        let (mut sum, mut amp, mut norm2, mut freq) = (0.0f32, 1.0f32, 0.0f32, 1.0f32);
        for i in 0..=self.depth.min(MAX_DEPTH) {
            sum += gradient(x * freq, y + i as f32 * 7.13) * amp;
            norm2 += amp * amp;
            amp *= 0.5;
            freq *= 2.0;
        }
        // Root-sum-square keeps the spread the same at every depth (a plain
        // sum shrank depth 4 to 60% of depth 0), and tanh lets the peaks
        // reach ± strength / 2 without ever flat-topping past it.
        (sum / norm2.sqrt() * 2.0).tanh() * 0.5 * self.strength
    }
}

/// 2-d gradient noise, roughly -1..1.
fn gradient(x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (ix, iy) = (x0 as i32, y0 as i32);
    let g = |cx: i32, cy: i32, dx: f32, dy: f32| {
        let a = hash(cx, cy) as f32 / u32::MAX as f32 * TAU;
        a.cos() * dx + a.sin() * dy
    };
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(fx), fade(fy));
    let a = g(ix, iy, fx, fy);
    let b = g(ix + 1, iy, fx - 1.0, fy);
    let c = g(ix, iy + 1, fx, fy - 1.0);
    let d = g(ix + 1, iy + 1, fx - 1.0, fy - 1.0);
    let top = a + (b - a) * u;
    let bot = c + (d - c) * u;
    // A 2-d gradient noise peaks near ±0.7; stretch it to about ±1.
    (top + (bot - top) * v) * 1.41
}

fn hash(x: i32, y: i32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^ (h >> 15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_wiggles_within_strength_and_is_smooth() {
        let n = CurveNoise { strength: 4.0, ..Default::default() };
        let v: Vec<f32> = (0..2000).map(|i| n.at(i as f32 * 0.25)).collect();
        assert!(v.iter().all(|x| x.abs() <= 2.0 + 1e-5), "outside ± strength / 2");
        let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(a, b), x| (a.min(*x), b.max(*x)));
        // It uses the range it says it has, and keeps its spread with depth.
        assert!(hi - lo > 3.0, "barely moves: {lo}..{hi}");
        let sd = |c: &CurveNoise| {
            ((0..4000).map(|i| c.at(i as f32 * 0.37).powi(2)).sum::<f32>() / 4000.0).sqrt()
        };
        let deep = CurveNoise { depth: 4, ..n.clone() };
        assert!((sd(&deep) / sd(&n) - 1.0).abs() < 0.15, "depth shrank it: {} vs {}", sd(&deep), sd(&n));
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.4, "not centred: {mean}");
        // Depth 0 is smooth: a quarter frame never jumps far at scale 10.
        let jump = v.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        assert!(jump < n.strength * 0.08, "jumps {jump}");
        // No dead spots on the lattice.
        assert!((0..20).any(|k| n.at(k as f32 * 10.0).abs() > 0.1));
    }

    #[test]
    fn phase_offset_scale_and_depth_do_what_they_say() {
        let n = CurveNoise::default();
        let shifted = CurveNoise { offset: 7.0, ..n.clone() };
        assert!((shifted.at(17.0) - n.at(10.0)).abs() < 1e-6, "offset slides it in time");
        let other = CurveNoise { phase: 5.0, ..n.clone() };
        assert!((0..50).any(|i| (other.at(i as f32) - n.at(i as f32)).abs() > 0.05), "phase reseeds");
        let slow = CurveNoise { scale: 20.0, ..n.clone() };
        assert!((slow.at(20.0) - n.at(10.0)).abs() < 1e-6, "scale stretches time");
        let rough = CurveNoise { depth: 4, ..n.clone() };
        // Roughness: the second difference, which the fine octaves own.
        let wig = |c: &CurveNoise| {
            (1..400)
                .map(|i| {
                    let t = i as f32 * 0.5;
                    (c.at(t + 0.5) - 2.0 * c.at(t) + c.at(t - 0.5)).abs()
                })
                .sum::<f32>()
        };
        assert!(wig(&rough) > wig(&n) * 1.2, "depth adds detail");
    }

    #[test]
    fn it_sizes_itself_to_the_curve() {
        assert_eq!(CurveNoise::sized_for([0.0, 500.0].into_iter()).strength, 50.0);
        assert!((CurveNoise::sized_for([0.8].into_iter()).strength - 0.08).abs() < 1e-6);
        assert_eq!(CurveNoise::sized_for([0.0].into_iter()).strength, 1.0);
    }
}
