//! Band ranges: an axis cut into N bands whose boundaries are SHARED.
//!
//! A classic colour-correct node divides the luminance axis into shadows, midtones and
//! highlights with a ranges widget, and the thing that makes it work is that
//! there is no such object as "the midtones' lower end" — there is one
//! boundary, and it belongs to the shadows and the midtones at once. Move it
//! and both follow. You cannot open a gap, you cannot make an overlap, and
//! the weights always sum to 1.
//!
//! This is that, with the count unfixed. It replaces an earlier model where
//! each band carried its own independent ends (see `LumaBand`'s history):
//! independent ends can express things a partition cannot, but they make the
//! common case — "move the line between these two" — into two edits that
//! have to agree, and they make the picture unreadable because nothing tells
//! you where a band stops.
//!
//! An [`Edge`] is that shared boundary: a position and a crossfade width.
//! Over the crossfade one band falls 1 → 0 while its neighbour rises 0 → 1,
//! by the same curve, so the pair sums to 1 everywhere. Widths are clamped
//! so two crossfades can never overlap, which is what keeps that true for
//! the whole axis rather than just for one pair.

/// How a crossfade gets from one band to the next.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BandCurve {
    /// Straight ramp. Predictable, and what you want when the bands are
    /// standing in for a hard selection.
    Linear,
    /// Smoothstep. Flat where it meets each band, so a correction fades in
    /// without a visible crease at the boundary.
    #[default]
    Smooth,
}

impl BandCurve {
    pub const ALL: [BandCurve; 2] = [BandCurve::Linear, BandCurve::Smooth];

    pub fn name(self) -> &'static str {
        match self {
            BandCurve::Linear => "linear",
            BandCurve::Smooth => "smooth",
        }
    }

    /// Rising side of a crossfade, `t` already normalized to 0..1.
    #[inline]
    pub fn eval(self, t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        match self {
            BandCurve::Linear => t,
            BandCurve::Smooth => t * t * (3.0 - 2.0 * t),
        }
    }
}

/// One shared boundary between two adjacent bands.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Edge {
    /// Where the two bands are equal, in axis units.
    pub at: f32,
    /// Total width of the crossfade, centred on `at`. 0 is a hard cut.
    pub width: f32,
}

impl Edge {
    pub fn new(at: f32, width: f32) -> Self {
        Self { at, width }
    }
}

/// The partition itself.
///
/// On a **wrapping** axis (hue) there are as many bands as edges, and band
/// `i` runs from edge `i-1` to edge `i` the long way round the circle. On a
/// **linear** axis (luma) there is one more band than edge, because the two
/// ends of the axis are not joined.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BandRanges {
    /// Sorted ascending. Kept sorted by every mutator here.
    pub edges: Vec<Edge>,
    pub curve: BandCurve,
    /// The axis itself: `(lo, hi)`. A wrapping axis treats `hi` and `lo` as
    /// the same place.
    pub span: (f32, f32),
    pub wrap: bool,
}

impl BandRanges {
    /// `n` bands evenly spaced across the span, each boundary soft by
    /// `soft` (a fraction of the spacing, 0..1).
    pub fn even(n: usize, span: (f32, f32), wrap: bool, soft: f32) -> Self {
        let n = n.max(1);
        let (lo, hi) = span;
        let len = (hi - lo).abs().max(1e-6);
        // One band on a wrapping axis has NO boundary — the circle is
        // already closed. Giving it one makes the band its own neighbour.
        let count = if wrap {
            if n <= 1 {
                0
            } else {
                n
            }
        } else {
            n.saturating_sub(1)
        };
        let step = len / n as f32;
        let width = (step * soft.clamp(0.0, 1.0)).max(0.0);
        let edges = (0..count)
            .map(|i| {
                // A wrapping axis puts the first boundary half a step in, so
                // band 0 is CENTRED on the span's start — hue 0 is red, and
                // "red" means the wedge around red, not the wedge after it.
                let t = if wrap { i as f32 + 0.5 } else { i as f32 + 1.0 };
                Edge::new(lo + step * t, width)
            })
            .collect();
        Self { edges, curve: BandCurve::default(), span, wrap }
    }

    pub fn band_count(&self) -> usize {
        if self.wrap {
            self.edges.len().max(1)
        } else {
            self.edges.len() + 1
        }
    }

    pub fn axis_len(&self) -> f32 {
        (self.span.1 - self.span.0).abs().max(1e-6)
    }

    /// Distance from edge `i` to edge `i+1`, wrapping if the axis does.
    fn gap(&self, i: usize) -> f32 {
        let n = self.edges.len();
        if n == 0 {
            return self.axis_len();
        }
        if i + 1 < n {
            self.edges[i + 1].at - self.edges[i].at
        } else if self.wrap {
            self.edges[0].at + self.axis_len() - self.edges[n - 1].at
        } else {
            self.span.1 - self.edges[n - 1].at
        }
    }

    fn gap_before(&self, i: usize) -> f32 {
        let n = self.edges.len();
        if i > 0 {
            self.edges[i].at - self.edges[i - 1].at
        } else if self.wrap && n > 0 {
            self.edges[0].at + self.axis_len() - self.edges[n - 1].at
        } else {
            self.edges[i].at - self.span.0
        }
    }

    /// Keep the invariant that makes the weights sum to 1: edges sorted, and
    /// no crossfade reaching past its neighbour's centre. Without the clamp
    /// three bands can be live at one point and the partition stops being
    /// one — visible immediately as a band that never reaches full strength.
    pub fn normalize(&mut self) {
        self.edges
            .sort_by(|a, b| a.at.partial_cmp(&b.at).unwrap_or(std::cmp::Ordering::Equal));
        if !self.wrap {
            let (lo, hi) = self.span;
            for e in &mut self.edges {
                e.at = e.at.clamp(lo, hi);
            }
        }
        // Two adjacent crossfades must not overlap: half of each has to fit
        // in the gap between them, so clamping each width to the whole gap
        // is a safe (and simple) sufficient condition.
        //
        // Against the END OF THE AXIS the constraint is different and looser:
        // there is no neighbouring crossfade there, only the requirement that
        // the shoulder not fall off the axis — a HALF width, not a whole one.
        // Treating the ends like neighbours squeezed every end band, and a
        // migrated threshold key came back with a visibly narrower ramp than
        // the one it was authored with.
        let n = self.edges.len();
        for i in 0..n {
            let before = if i > 0 || self.wrap {
                self.gap_before(i)
            } else {
                2.0 * self.gap_before(i)
            };
            let after = if i + 1 < n || self.wrap {
                self.gap(i)
            } else {
                2.0 * self.gap(i)
            };
            let room = before.min(after).max(0.0);
            self.edges[i].width = self.edges[i].width.clamp(0.0, room);
        }
    }

    /// Signed distance from `x` to edge `i`, on the axis and wrapping if it
    /// does — positive means `x` is on the higher-band side.
    fn delta(&self, i: usize, x: f32) -> f32 {
        let mut d = x - self.edges[i].at;
        if self.wrap {
            let len = self.axis_len();
            while d > len * 0.5 {
                d -= len;
            }
            while d < -len * 0.5 {
                d += len;
            }
        }
        d
    }

    /// Each band's weight at `x`. Always sums to 1 (within float error), and
    /// at most two entries are ever non-zero.
    pub fn weights(&self, x: f32) -> Vec<f32> {
        let n = self.band_count();
        let mut w = vec![0.0; n];
        if self.edges.is_empty() {
            w[0] = 1.0;
            return w;
        }
        // Which band `x` sits in, before any crossfade: the number of edges
        // it is past.
        let band = if self.wrap {
            // Band i is bounded below by edge i-1 and above by edge i, so
            // being before edge 0 (or past the last) is band 0.
            self.edges.iter().position(|e| x < e.at).unwrap_or(0)
        } else {
            self.edges.iter().filter(|e| x >= e.at).count()
        };
        w[band.min(n - 1)] = 1.0;
        // Then the one crossfade `x` can be inside.
        for i in 0..self.edges.len() {
            let e = self.edges[i];
            if e.width <= 0.0 {
                continue;
            }
            let d = self.delta(i, x);
            if d.abs() * 2.0 >= e.width {
                continue;
            }
            let t = self.curve.eval(d / e.width + 0.5);
            // Edge i separates band i (below) from band i+1 (above); on a
            // wrapping axis the last edge closes back onto band 0.
            let (below, above) = if self.wrap {
                (i, (i + 1) % n)
            } else {
                (i, i + 1)
            };
            if below == above {
                // A single wrapping band: the boundary has the same band on
                // both sides, so there is nothing to fade between.
                continue;
            }
            w.iter_mut().for_each(|v| *v = 0.0);
            w[below.min(n - 1)] = 1.0 - t;
            w[above.min(n - 1)] = t;
            break;
        }
        w
    }

    /// Add a boundary at `at`, splitting whichever band contains it. The new
    /// edge inherits its neighbours' softness so the picture does not jump.
    pub fn split_at(&mut self, at: f32) {
        let width = self
            .edges
            .first()
            .map(|e| e.width)
            .unwrap_or(self.axis_len() * 0.1);
        self.edges.push(Edge::new(at, width));
        self.normalize();
    }

    /// Remove band `i` by dissolving the boundary that ends it. The two
    /// neighbours become one band — the only thing removing a band CAN mean
    /// once the boundaries are shared.
    pub fn remove_band(&mut self, i: usize) {
        if self.edges.is_empty() || self.band_count() <= 1 {
            return;
        }
        let e = if self.wrap { i % self.edges.len() } else { i.min(self.edges.len() - 1) };
        self.edges.remove(e);
        self.normalize();
    }

    /// Move edge `i` to `at`, carrying its crossfade. This is the locked
    /// gesture: one call, two bands changed.
    pub fn move_edge(&mut self, i: usize, at: f32) {
        if i >= self.edges.len() {
            return;
        }
        let mut at = at;
        if self.wrap {
            let len = self.axis_len();
            at = self.span.0 + (at - self.span.0).rem_euclid(len);
        } else {
            // Keep the order without swapping under the cursor: an edge may
            // reach its neighbours but not pass them.
            let lo = if i == 0 { self.span.0 } else { self.edges[i - 1].at };
            let hi = if i + 1 == self.edges.len() {
                self.span.1
            } else {
                self.edges[i + 1].at
            };
            at = at.clamp(lo, hi);
        }
        self.edges[i].at = at;
        self.normalize();
    }

    /// The centre of band `i` — where the band is at full strength and the
    /// natural place to hang its swatch or its label.
    pub fn band_center(&self, i: usize) -> f32 {
        let n = self.band_count();
        if self.edges.is_empty() {
            return 0.5 * (self.span.0 + self.span.1);
        }
        if self.wrap {
            let m = self.edges.len();
            let lo = self.edges[(i + m - 1) % m].at;
            let mut hi = self.edges[i % m].at;
            if hi < lo {
                hi += self.axis_len();
            }
            let c = 0.5 * (lo + hi);
            self.span.0 + (c - self.span.0).rem_euclid(self.axis_len())
        } else {
            let lo = if i == 0 { self.span.0 } else { self.edges[i - 1].at };
            let hi = if i >= self.edges.len() { self.span.1 } else { self.edges[i].at };
            let _ = n;
            0.5 * (lo + hi)
        }
    }
}

/// Band `i`'s response as control points, ready to hand to a curve-shaped
/// consumer (a `LumaBand`, a shader's packed curve).
///
/// This is the bridge between the partition and everything that renders a
/// band. A middle band is a plateau: it rises across the boundary below it,
/// sits at 1, and falls across the boundary above it — which is exactly the
/// shape a pair of independent thresholds could never express, and the
/// reason `LumaBand` grew explicit end coverages.
///
/// The points are the crossfade CORNERS, not a resampling: four points for a
/// middle band, two or three at the ends. Whatever interpolates them (the
/// shader's `curve_eval`, the UI's preview) decides linear versus smooth,
/// and [`BandRanges::curve`] says which it should be.
pub fn band_curve(r: &BandRanges, i: usize) -> Vec<[f32; 2]> {
    let n = r.band_count();
    let (lo, hi) = r.span;
    if r.edges.is_empty() || n <= 1 {
        return vec![[lo, 1.0], [hi, 1.0]];
    }
    let i = i.min(n - 1);
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(4);
    // The boundary BELOW this band (the one it rises across), and the one
    // above (the one it falls across). On a wrapping axis both always exist;
    // on a linear axis the first and last bands are missing one.
    let below = if r.wrap {
        Some(r.edges[(i + r.edges.len() - 1) % r.edges.len()])
    } else if i > 0 {
        Some(r.edges[i - 1])
    } else {
        None
    };
    let above = if r.wrap {
        Some(r.edges[i % r.edges.len()])
    } else if i < r.edges.len() {
        Some(r.edges[i])
    } else {
        None
    };
    match below {
        Some(e) => {
            out.push([e.at - e.width * 0.5, 0.0]);
            out.push([e.at + e.width * 0.5, 1.0]);
        }
        None => out.push([lo, 1.0]),
    }
    match above {
        Some(e) => {
            out.push([e.at - e.width * 0.5, 1.0]);
            out.push([e.at + e.width * 0.5, 0.0]);
        }
        None => out.push([hi, 1.0]),
    }
    // A hard boundary puts two points on the same x; nudge so the curve
    // stays a function (and so a consumer that sorts by x keeps the order).
    for k in 1..out.len() {
        if out[k][0] <= out[k - 1][0] {
            out[k][0] = out[k - 1][0] + 1e-5;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum_is_one(b: &BandRanges) {
        let (lo, hi) = b.span;
        for k in 0..=400 {
            let x = lo + (hi - lo) * k as f32 / 400.0;
            let s: f32 = b.weights(x).iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "weights sum to {s} at x={x}");
        }
    }

    #[test]
    fn a_partition_always_sums_to_one() {
        // The property the whole model exists for: no gaps, no overlaps, on
        // both kinds of axis and both curves.
        for wrap in [false, true] {
            for curve in BandCurve::ALL {
                for n in [1usize, 2, 3, 6, 9] {
                    let mut b = BandRanges::even(n, (0.0, 1.0), wrap, 0.6);
                    b.curve = curve;
                    sum_is_one(&b);
                }
            }
        }
    }

    #[test]
    fn at_most_two_bands_are_ever_live() {
        let b = BandRanges::even(6, (0.0, 1.0), true, 1.0);
        for k in 0..=300 {
            let x = k as f32 / 300.0;
            let live = b.weights(x).iter().filter(|w| **w > 1e-5).count();
            assert!(live <= 2, "{live} bands live at {x}");
        }
    }

    #[test]
    fn moving_an_edge_moves_both_neighbours() {
        // The lock, stated as a test: one call, and the band on each side of
        // the boundary changes. Under the old independent-ends model this
        // took two edits that could disagree.
        let mut b = BandRanges::even(3, (0.0, 1.0), false, 0.0);
        let before = (b.weights(0.3)[0], b.weights(0.3)[1]);
        assert_eq!(before, (1.0, 0.0));
        b.move_edge(0, 0.2); // was at 1/3
        let after = (b.weights(0.3)[0], b.weights(0.3)[1]);
        assert_eq!(after, (0.0, 1.0), "the point changed hands, not fell in a gap");
        sum_is_one(&b);
    }

    #[test]
    fn an_edge_cannot_pass_its_neighbour() {
        let mut b = BandRanges::even(3, (0.0, 1.0), false, 0.0);
        b.move_edge(1, -5.0);
        assert!(b.edges[1].at >= b.edges[0].at, "{:?}", b.edges);
        b.move_edge(0, 5.0);
        assert!(b.edges[0].at <= b.edges[1].at, "{:?}", b.edges);
        sum_is_one(&b);
    }

    #[test]
    fn crossfades_are_clamped_so_they_cannot_overlap() {
        // Ask for absurd softness on tightly packed edges; normalize must
        // pull it back rather than let three bands be live at once.
        let mut b = BandRanges::even(8, (0.0, 1.0), true, 1.0);
        for e in &mut b.edges {
            e.width = 99.0;
        }
        b.normalize();
        sum_is_one(&b);
        for k in 0..=300 {
            let live = b.weights(k as f32 / 300.0).iter().filter(|w| **w > 1e-5).count();
            assert!(live <= 2);
        }
    }

    #[test]
    fn hue_wraps_so_the_last_band_meets_the_first() {
        let b = BandRanges::even(6, (0.0, 1.0), true, 0.5);
        // Just below 0 and just above 0 are the same band on a hue wheel.
        let a = b.weights(0.999);
        let c = b.weights(0.001);
        let hot = |w: &Vec<f32>| {
            w.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0
        };
        assert_eq!(hot(&a), hot(&c), "the wheel is not joined: {a:?} vs {c:?}");
        sum_is_one(&b);
    }

    #[test]
    fn band_zero_is_centred_on_the_span_start_when_wrapping() {
        // "red" has to mean the wedge AROUND red, not the wedge after it.
        let b = BandRanges::even(6, (0.0, 1.0), true, 0.0);
        assert!(b.weights(0.0)[0] > 0.99, "{:?}", b.weights(0.0));
        assert!((b.band_center(0)).min(1.0 - b.band_center(0)) < 1e-4);
    }

    #[test]
    fn a_hard_edge_is_a_step_and_a_soft_one_is_not() {
        let mut b = BandRanges::even(2, (0.0, 1.0), false, 0.0);
        assert_eq!(b.weights(0.499)[0], 1.0);
        assert_eq!(b.weights(0.501)[1], 1.0);
        b.edges[0].width = 0.4;
        b.normalize();
        let w = b.weights(0.5);
        assert!((w[0] - 0.5).abs() < 1e-4 && (w[1] - 0.5).abs() < 1e-4, "{w:?}");
        assert!(b.weights(0.45)[0] > 0.5 && b.weights(0.55)[1] > 0.5);
    }

    #[test]
    fn splitting_adds_a_band_and_removing_dissolves_one() {
        let mut b = BandRanges::even(3, (0.0, 1.0), false, 0.2);
        assert_eq!(b.band_count(), 3);
        b.split_at(0.5);
        assert_eq!(b.band_count(), 4);
        sum_is_one(&b);
        b.remove_band(1);
        assert_eq!(b.band_count(), 3);
        sum_is_one(&b);
        // A single band cannot be removed — there has to be something left.
        let mut one = BandRanges::even(1, (0.0, 1.0), false, 0.0);
        one.remove_band(0);
        assert_eq!(one.band_count(), 1);
    }

    #[test]
    fn linear_and_smooth_agree_at_the_ends_and_differ_between() {
        let mut lin = BandRanges::even(2, (0.0, 1.0), false, 1.0);
        lin.curve = BandCurve::Linear;
        let mut sm = lin.clone();
        sm.curve = BandCurve::Smooth;
        assert!((lin.weights(0.5)[1] - sm.weights(0.5)[1]).abs() < 1e-5);
        let q = 0.5 - lin.edges[0].width * 0.25;
        assert!(
            (lin.weights(q)[1] - sm.weights(q)[1]).abs() > 0.02,
            "the two curves should not be the same shape"
        );
    }

    #[test]
    fn a_middle_band_is_a_plateau_not_a_threshold() {
        // The shape the old two-threshold model could not express.
        let r = BandRanges::even(3, (0.0, 1.0), false, 0.6);
        let mid = band_curve(&r, 1);
        assert_eq!(mid.len(), 4, "{mid:?}");
        assert_eq!(mid[0][1], 0.0);
        assert_eq!(mid[1][1], 1.0);
        assert_eq!(mid[2][1], 1.0);
        assert_eq!(mid[3][1], 0.0);
        assert!(mid[0][0] < mid[1][0] && mid[1][0] < mid[2][0] && mid[2][0] < mid[3][0]);
    }

    #[test]
    fn the_end_bands_reach_the_end_of_the_axis() {
        let r = BandRanges::even(3, (0.0, 2.0), false, 0.5);
        let first = band_curve(&r, 0);
        let last = band_curve(&r, 2);
        assert_eq!(first[0], [0.0, 1.0], "shadows must start solid at black");
        assert_eq!(*last.last().unwrap(), [2.0, 1.0], "highlights must reach the top");
    }

    #[test]
    fn every_wrapping_band_is_a_plateau() {
        let r = BandRanges::even(6, (0.0, 1.0), true, 0.5);
        for i in 0..r.band_count() {
            let c = band_curve(&r, i);
            assert_eq!(c.len(), 4, "band {i}: {c:?}");
            assert_eq!((c[0][1], c[3][1]), (0.0, 0.0));
        }
    }

    #[test]
    fn the_derived_curves_are_the_weights() {
        // band_curve must describe the SAME partition weights() does, or the
        // picture and the render disagree.
        let mut r = BandRanges::even(4, (0.0, 1.0), false, 0.7);
        r.curve = BandCurve::Linear;
        for i in 0..r.band_count() {
            let c = band_curve(&r, i);
            for k in 0..=60 {
                let x = k as f32 / 60.0;
                let want = r.weights(x)[i];
                // Piecewise-linear lookup through the derived points.
                let got = if x <= c[0][0] {
                    c[0][1]
                } else if x >= c[c.len() - 1][0] {
                    c[c.len() - 1][1]
                } else {
                    let mut v = c[0][1];
                    for w in c.windows(2) {
                        if x >= w[0][0] && x <= w[1][0] {
                            let t = (x - w[0][0]) / (w[1][0] - w[0][0]).max(1e-9);
                            v = w[0][1] + (w[1][1] - w[0][1]) * t;
                            break;
                        }
                    }
                    v
                };
                assert!((got - want).abs() < 1e-3, "band {i} at {x}: {got} vs {want}");
            }
        }
    }

    #[test]
    fn a_hard_boundary_still_yields_a_function() {
        let r = BandRanges::even(3, (0.0, 1.0), false, 0.0);
        for i in 0..3 {
            let c = band_curve(&r, i);
            for w in c.windows(2) {
                assert!(w[1][0] > w[0][0], "band {i} has a vertical segment: {c:?}");
            }
        }
    }
}
