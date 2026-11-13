//! Single-point pattern tracking: the Tracker node's PATTERN mode.
//!
//! A point tracker in the usual sense is a small square PATTERN cut
//! around the point on the reference frame, aligned into every later frame
//! to subpixel precision. That is exactly the planar tracker's solve on a
//! square region with a small model, so this module is a thin wrapper over
//! `planar::Tracker` rather than a second aligner: the same ZNCC score, the
//! same robust weights (an occluder is weighted out, not averaged in), the
//! same trust rule (a lost frame COASTS on its velocity and is never cut
//! into the template) and the same drift-only re-anchor.
//!
//! What a point adds is a search REACH: a planar region is big, so a
//! frame's motion is small against it and the pyramid covers it. A pattern
//! is a few dozen pixels, often smaller than one frame's travel, so the
//! re-acquiring coarse search is told how far to look (`PointOpts.search`).
//!
//! Why this and not a learned tracker: the TAP family (TAPIR, BootsTAPIR,
//! TAPNext) is robust but runs at 256² — a pixel there is eight on a 2K
//! plate — and comp work wants tenths of a pixel. A learned model is the
//! right RE-ACQUISITION after a loss; the precision stays here.

use crate::planar::{h_apply, Lost, Model, Plane, Pyramid, TrackOpts, Tracker};

/// The trust bar follows the pattern's own recent scores: a fit this far
/// under them has to land where the velocity put it. A FIXED bar is wrong
/// either way. At the region's 0.85 a covered 32 px pattern took a
/// lookalike 35 px away at 0.852 (its honest scores were 1.0); at 0.95 the
/// hcr080 necklace, whose honest scores slide from 0.97 to 0.86 as the
/// camera swings round, had every fit from frame 13 refused and the track
/// held still for the rest of the shot.
const TRUST_MARGIN: f32 = 0.06;
const TRUST_FLOOR: f32 = 0.85;
const TRUST_CEIL: f32 = 0.95;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointOpts {
    /// What the pattern may do between frames; the POINT is the pattern's
    /// centre carried by the fit either way. Similarity is the default
    /// because translation alone cannot absorb a pattern that TURNS: the
    /// match stays high while the fit slides (`probe_sizes`: 16° over 40
    /// frames, 8.6 px worst translation-only, 0.17 px similarity, and no
    /// re-anchor threshold rescues it). Translation is for footage that
    /// genuinely only moves, where it is the stiffer fit.
    pub model: Model,
    /// How far the point may travel between frames before the coarse search
    /// can no longer find it again, in full-res pixels.
    pub search: f64,
    /// Re-cut the pattern once the match slides under this score.
    pub reanchor_below: f32,
}

impl Default for PointOpts {
    fn default() -> Self {
        Self { model: Model::Similarity, search: 64.0, reanchor_below: 0.8 }
    }
}

/// One frame's answer.
#[derive(Clone, Copy, Debug)]
pub struct PointResult {
    /// The point in this frame, pixels (centres at +0.5).
    pub pos: [f64; 2],
    pub score: f32,
    /// No trusted fit: `pos` is the coast (or the hold).
    pub lost: bool,
    pub why: Option<Lost>,
    pub reanchored: bool,
}

pub struct PointTracker {
    inner: Tracker,
    point: [f64; 2],
    /// Smoothed score of the trusted fits so far.
    recent: Option<f32>,
}

impl PointTracker {
    /// Start on `reference`: the point at `center`, the pattern the box
    /// `center ± half` (pixels).
    pub fn new(reference: Plane, center: [f64; 2], half: [f64; 2], opts: PointOpts) -> Self {
        let (hx, hy) = (half[0].max(4.0), half[1].max(4.0));
        let [cx, cy] = center;
        let square = vec![[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]];
        let topts = TrackOpts {
            model: opts.model,
            reanchor_below: opts.reanchor_below,
            search: opts.search.max(0.0),
            confident: TRUST_FLOOR,
            ..Default::default()
        };
        Self { inner: Tracker::new(smooth(reference), square, topts), point: center, recent: None }
    }

    /// Track one more frame (frames in order, either direction).
    pub fn step(&mut self, frame: Plane) -> PointResult {
        let pyr = self.prepare(frame);
        self.step_prepared(&pyr)
    }

    /// The frame as the aligner reads it (smoothed, then a pyramid). Every
    /// point tracker on the same plate makes the same one, so a multi-point
    /// track prepares each frame ONCE and steps every point on it.
    pub fn prepare(&self, frame: Plane) -> Pyramid {
        self.inner.pyramid(smooth(frame))
    }

    pub fn step_prepared(&mut self, pyr: &Pyramid) -> PointResult {
        if let Some(recent) = self.recent {
            self.inner.set_confident((recent - TRUST_MARGIN).clamp(TRUST_FLOOR, TRUST_CEIL));
        }
        let r = self.inner.step_pyramid(pyr);
        if !r.lost {
            self.recent = Some(match self.recent {
                Some(s) => 0.5 * s + 0.5 * r.score,
                None => r.score,
            });
        }
        let (x, y) = h_apply(&r.h, self.point[0], self.point[1]);
        PointResult { pos: [x, y], score: r.score, lost: r.lost, why: r.why, reanchored: r.reanchored }
    }
}

/// A separable [1 4 6 4 1]/16 blur (σ ≈ 1 px). Grain is uncorrelated from
/// pixel to pixel and a pattern is small, so without it the fit chases
/// noise; the pattern's own structure is broader than this and keeps its
/// centre (a symmetric kernel moves no edge).
fn smooth(p: Plane) -> Plane {
    use rayon::prelude::*;
    const K: [f32; 5] = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0];
    let (w, h) = (p.w, p.h);
    if w < 5 || h < 5 {
        return p;
    }
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &p.px[y * w..(y + 1) * w];
        for (x, o) in out.iter_mut().enumerate() {
            *o = K.iter().enumerate().map(|(k, c)| c * row[(x + k).saturating_sub(2).min(w - 1)]).sum();
        }
    });
    let mut px = vec![0.0f32; w * h];
    px.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        for (x, o) in out.iter_mut().enumerate() {
            *o = K.iter().enumerate().map(|(k, c)| c * tmp[(y + k).saturating_sub(2).min(h - 1) * w + x]).sum();
        }
    });
    Plane::new(w, h, px)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planar::synth::{path, render, texture, view, RenderOpts};
    use crate::planar::{h_mul, H};

    const W: usize = 512;
    const HT: usize = 384;

    fn shot(t: usize, o: &RenderOpts, extra: &H) -> (Plane, H) {
        let tex = texture(W, HT, 11);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let hv = h_mul(extra, &h_mul(&path(W, HT, t), &base));
        let truth = h_mul(extra, &path(W, HT, t));
        (render(&tex, &hv, W, HT, o), truth)
    }

    /// Track `frames` frames of the synthetic path and return the worst
    /// and mean error of the tracked point against the truth, plus the
    /// number of frames lost.
    fn run(opts: PointOpts, half: f64, frames: usize, noise: f32, occl: bool) -> (f64, f64, usize) {
        let c = [W as f64 * 0.47, HT as f64 * 0.52];
        let o0 = RenderOpts { noise, ..Default::default() };
        let id = crate::planar::IDENTITY;
        let mut tr = PointTracker::new(shot(0, &o0, &id).0, c, [half, half], opts);
        let (mut worst, mut sum, mut lost) = (0.0f64, 0.0f64, 0usize);
        let mut scored = 0usize;
        for t in 1..frames {
            let (tx, ty) = h_apply(&path(W, HT, t), c[0], c[1]);
            // The occluder crosses the point over frames 12..18.
            let occluder = (occl && (12..18).contains(&t)).then_some(([tx, ty], half * 1.4));
            let o = RenderOpts { noise, seed: t as u32 + 1, occluder, ..Default::default() };
            let r = tr.step(shot(t, &o, &id).0);
            if r.lost {
                lost += 1;
            }
            let e = ((r.pos[0] - tx).powi(2) + (r.pos[1] - ty).powi(2)).sqrt();
            if std::env::var_os("WOBBLE_POINT_TRACE").is_some() {
                eprintln!("t {t:2} err {e:7.3} score {:.3} lost {} {:?} re {}", r.score, r.lost, r.why, r.reanchored);
            }
            // Frames under the occluder are judged on whether the track
            // comes BACK, not on the coast.
            if occl && (12..20).contains(&t) {
                continue;
            }
            worst = worst.max(e);
            sum += e;
            scored += 1;
        }
        (worst, sum / scored.max(1) as f64, lost)
    }

    /// The error of a pattern tracked over the synthetic path, pattern
    /// size by model. `cargo test -p engine --release probe_sizes --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn probe_sizes() {
        for noise in [0.0f32, 0.02, 0.08] {
            for half in [8.0, 16.0, 32.0] {
                for model in [Model::Translation, Model::Similarity] {
                    let opts = PointOpts { model, ..Default::default() };
                    let (worst, mean, lost) = run(opts, half, 40, noise, false);
                    eprintln!(
                        "grain {noise:.2} half {half:>4} {:>11}: worst {worst:.3} mean {mean:.3} lost {lost}",
                        model.label()
                    );
                }
            }
        }
    }

    #[test]
    fn a_textured_point_tracks_to_a_fraction_of_a_pixel() {
        // The synthetic path: translation, 0.4°/frame rotation, 0.6%/frame
        // growth and a little perspective — 16° and +24% by the end.
        let (worst, mean, lost) = run(PointOpts::default(), 16.0, 40, 0.0, false);
        eprintln!("similarity: worst {worst:.3} px, mean {mean:.3} px, lost {lost}");
        assert_eq!(lost, 0);
        assert!(worst < 0.3 && mean < 0.15, "worst {worst} mean {mean}");
    }

    #[test]
    fn translation_holds_subpixel_steps_on_footage_that_only_moves() {
        let tex = texture(W, HT, 3);
        let c = [W as f64 * 0.5, HT as f64 * 0.5];
        let at = |t: usize| {
            let (dx, dy) = (1.37 * t as f64, -0.61 * t as f64);
            let v = view(W, HT, 0.0, 1.0, dx, dy, 0.0, 0.0);
            (render(&tex, &v, W, HT, &RenderOpts::default()), [c[0] + dx, c[1] + dy])
        };
        let opts = PointOpts { model: Model::Translation, ..Default::default() };
        let mut tr = PointTracker::new(at(0).0, c, [16.0, 16.0], opts);
        let mut worst = 0.0f64;
        for t in 1..30 {
            let (f, truth) = at(t);
            let r = tr.step(f);
            assert!(!r.lost, "frame {t}: {r:?}");
            worst = worst.max(((r.pos[0] - truth[0]).powi(2) + (r.pos[1] - truth[1]).powi(2)).sqrt());
        }
        eprintln!("translation, pure motion: worst {worst:.3} px");
        assert!(worst < 0.1, "worst {worst}");
    }

    /// Grain against pattern size: light grain on a small pattern, heavy
    /// grain on a bigger one. Heavy grain on a SMALL pattern of this
    /// low-contrast texture is not held (2 px worst at half 16) — a pattern
    /// has to carry more structure than the noise, and `probe_sizes` shows
    /// where that line sits.
    #[test]
    fn grain_costs_little() {
        let (worst, mean, lost) = run(PointOpts::default(), 16.0, 40, 0.02, false);
        eprintln!("grain 0.02 half 16: worst {worst:.3} px, mean {mean:.3} px, lost {lost}");
        assert!(lost == 0 && worst < 0.7 && mean < 0.25, "worst {worst} mean {mean} lost {lost}");
        let (worst, mean, lost) = run(PointOpts::default(), 32.0, 40, 0.08, false);
        eprintln!("grain 0.08 half 32: worst {worst:.3} px, mean {mean:.3} px, lost {lost}");
        assert!(lost == 0 && worst < 0.5 && mean < 0.25, "worst {worst} mean {mean} lost {lost}");
    }

    #[test]
    fn an_occluder_over_the_point_coasts_and_the_track_comes_back() {
        let (worst, mean, lost) = run(PointOpts::default(), 16.0, 40, 0.0, true);
        eprintln!("occluded: worst {worst:.3} px after, mean {mean:.3} px, lost {lost}");
        assert!(lost >= 1, "the covered frames must be reported, not trusted");
        assert!(worst < 0.5, "worst {worst} once the occluder has passed");
    }

    #[test]
    fn a_jump_further_than_the_pattern_is_wide_is_found_by_the_search() {
        let c = [W as f64 * 0.4, HT as f64 * 0.5];
        let tex = texture(W, HT, 5);
        let at = |dx: f64| {
            let v = view(W, HT, 0.0, 1.0, dx, 0.0, 0.0, 0.0);
            render(&tex, &v, W, HT, &RenderOpts::default())
        };
        // 50 px in one frame with a 12 px pattern: four widths away.
        let near = PointOpts { model: Model::Translation, search: 0.0, ..Default::default() };
        let far = PointOpts { model: Model::Translation, search: 64.0, ..Default::default() };
        let mut a = PointTracker::new(at(0.0), c, [6.0, 6.0], far);
        let r = a.step(at(50.0));
        assert!(!r.lost && (r.pos[0] - c[0] - 50.0).abs() < 0.1, "{r:?}");
        // With no reach past the default the same jump is (honestly) lost.
        let mut b = PointTracker::new(at(0.0), c, [6.0, 6.0], near);
        let r = b.step(at(50.0));
        assert!(r.lost || (r.pos[0] - c[0] - 50.0).abs() < 0.1, "a wrong answer reported as found: {r:?}");
    }
}
