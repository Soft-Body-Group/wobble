//! Points to a plane motion: the solve behind a multi-point Tracker, and
//! anything else that moves a set of points and wants a picture to follow
//! (roto shapes riding on trackers are the next customer).
//!
//! Given where N points were at a reference frame and where they are now,
//! `solve` returns the reference → now homography, choosing the model the
//! point count can support:
//!
//! | points | model |
//! |---|---|
//! | 0 | identity |
//! | 1 | translation |
//! | 2, 3, 5+ | similarity (least squares: translate, rotate, uniform scale) |
//! | 4 | perspective (the corner pin through the four points) |
//!
//! Everything is in PIXELS, y down — rotation is only a rotation where the
//! axes share a unit. `pins` turns the result into a corner pin in the
//! normalised convention `Transform.pin` and CornerPin use.

use crate::planar::{h_apply, h_from_quad, h_translation, H, IDENTITY};

/// What `solve` does with `n` points, for a panel to say.
pub fn model_label(n: usize) -> &'static str {
    match n {
        0 => "nothing to solve",
        1 => "move",
        4 => "corner pin",
        _ => "move + turn + scale",
    }
}

/// Reference → now, from matched points (pixels). Mismatched lengths use
/// the common prefix.
pub fn solve(reference: &[[f64; 2]], now: &[[f64; 2]]) -> H {
    let n = reference.len().min(now.len());
    let (a, b) = (&reference[..n], &now[..n]);
    match n {
        0 => IDENTITY,
        1 => h_translation(b[0][0] - a[0][0], b[0][1] - a[0][1]),
        4 => {
            let src = [a[0], a[1], a[2], a[3]];
            let dst = [b[0], b[1], b[2], b[3]];
            // Three of the four in a line has no perspective answer; the
            // similarity still does.
            h_from_quad(&src, &dst).filter(|h| h.iter().all(|v| v.is_finite())).unwrap_or_else(|| similarity(a, b))
        }
        _ => similarity(a, b),
    }
}

/// Least-squares similarity (Procrustes): points as complex numbers about
/// their centroids, `q ≈ z·p` with `z = Σ conj(p)·q / Σ|p|²`.
pub fn similarity(a: &[[f64; 2]], b: &[[f64; 2]]) -> H {
    let n = a.len().min(b.len());
    if n == 0 {
        return IDENTITY;
    }
    let cen = |q: &[[f64; 2]]| {
        let (sx, sy) = q[..n].iter().fold((0.0, 0.0), |s, p| (s.0 + p[0], s.1 + p[1]));
        [sx / n as f64, sy / n as f64]
    };
    let (ca, cb) = (cen(a), cen(b));
    let (mut re, mut im, mut pp) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..n {
        let p = [a[k][0] - ca[0], a[k][1] - ca[1]];
        let q = [b[k][0] - cb[0], b[k][1] - cb[1]];
        re += p[0] * q[0] + p[1] * q[1];
        im += p[0] * q[1] - p[1] * q[0];
        pp += p[0] * p[0] + p[1] * p[1];
    }
    let (c, s) = if pp > 1e-9 { (re / pp, im / pp) } else { (1.0, 0.0) };
    // x' = c·x − s·y + tx, with the reference centroid landing on today's.
    let tx = cb[0] - (c * ca[0] - s * ca[1]);
    let ty = cb[1] - (s * ca[0] + c * ca[1]);
    [c, -s, tx, s, c, ty, 0.0, 0.0, 1.0]
}

/// The unit quad (the picture's corners, normalised) carried by `h`, in
/// the normalised pin convention. `size` is the picture in pixels.
pub fn pins(h: &H, size: [f64; 2]) -> [[f32; 2]; 4] {
    let (w, hh) = (size[0].max(1.0), size[1].max(1.0));
    let mut out = crate::graph::UNIT_QUAD;
    for (k, c) in crate::graph::UNIT_QUAD.iter().enumerate() {
        let (x, y) = h_apply(h, c[0] as f64 * w, c[1] as f64 * hh);
        out[k] = [(x / w) as f32, (y / hh) as f32];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planar::h_mul;

    fn map(h: &H, p: &[[f64; 2]]) -> Vec<[f64; 2]> {
        p.iter().map(|q| {
            let (x, y) = h_apply(h, q[0], q[1]);
            [x, y]
        }).collect()
    }

    fn close(a: &H, b: &H, pts: &[[f64; 2]]) -> f64 {
        map(a, pts).iter().zip(map(b, pts)).map(|(u, v)| (u[0] - v[0]).hypot(u[1] - v[1])).fold(0.0, f64::max)
    }

    const PTS: [[f64; 2]; 5] = [[100.0, 120.0], [700.0, 90.0], [680.0, 500.0], [130.0, 470.0], [400.0, 300.0]];

    #[test]
    fn each_count_recovers_the_motion_it_can_express() {
        let probe = [[0.0, 0.0], [800.0, 0.0], [800.0, 600.0], [0.0, 600.0]];
        // One point: a translation.
        let t = h_translation(12.5, -7.25);
        assert!(close(&solve(&PTS[..1], &map(&t, &PTS[..1])), &t, &probe) < 1e-9);
        // Two, three and five: a similarity, exactly.
        let (c, s) = (0.3f64.cos() * 1.2, 0.3f64.sin() * 1.2);
        let sim: H = [c, -s, 40.0, s, c, -15.0, 0.0, 0.0, 1.0];
        for n in [2, 3, 5] {
            let got = solve(&PTS[..n], &map(&sim, &PTS[..n]));
            assert!(close(&got, &sim, &probe) < 1e-6, "{n} points");
        }
        // Four: the full perspective.
        let persp: H = h_mul(&sim, &[1.0, 0.1, 0.0, -0.05, 0.95, 0.0, 2e-4, -1e-4, 1.0]);
        let got = solve(&PTS[..4], &map(&persp, &PTS[..4]));
        assert!(close(&got, &persp, &probe) < 1e-6);
    }

    #[test]
    fn degenerate_input_never_goes_non_finite() {
        let same = [[5.0, 5.0]; 4];
        let h = solve(&same, &same);
        assert!(h.iter().all(|v| v.is_finite()), "{h:?}");
        let line = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let h = solve(&line, &line);
        assert!(h.iter().all(|v| v.is_finite()), "{h:?}");
        assert_eq!(solve(&[], &[]), IDENTITY);
    }

    #[test]
    fn pins_at_rest_are_the_unit_quad() {
        assert_eq!(pins(&IDENTITY, [1920.0, 1080.0]), crate::graph::UNIT_QUAD);
        // A 10 px move on a 1000 px picture moves every pin by 0.01.
        let p = pins(&h_translation(10.0, 0.0), [1000.0, 500.0]);
        for (k, c) in crate::graph::UNIT_QUAD.iter().enumerate() {
            assert!((p[k][0] - c[0] - 0.01).abs() < 1e-6 && (p[k][1] - c[1]).abs() < 1e-6);
        }
    }
}
