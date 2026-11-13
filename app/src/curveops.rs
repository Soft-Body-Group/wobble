//! Operations over a curve's SELECTED keys — the curve editor's key menu,
//! pie and sliders. Pure: an `AnimCurve` and the frames of its selected
//! keys in, the curve changed in place (and the selected frames after the
//! op) out, so every one is unit-tested without a UI.
//!
//! The selection is a set of FRAMES (what `curves::KeyId` carries), and
//! "a run" is a stretch of consecutive selected keys: the slider ops work
//! run by run against the unselected keys either side of it, Blender's
//! "neighbours".

use crate::anim::{AnimCurve, EaseDir, Interp, Key};

/// Indices of the selected keys, in order.
fn picked(c: &AnimCurve, sel: &[f64]) -> Vec<usize> {
    let mut v: Vec<usize> = sel.iter().filter_map(|f| c.index_of(*f)).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Consecutive selected indices, as inclusive (first, last) pairs.
fn runs(idx: &[usize]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for &i in idx {
        match out.last_mut() {
            Some((_, b)) if *b + 1 == i => *b = i,
            _ => out.push((i, i)),
        }
    }
    out
}

/// Put keys back into a curve whose originals were taken out: a key that
/// lands on another key's frame REPLACES it (the dope sheet's rule).
pub fn place(c: &mut AnimCurve, keys: Vec<Key>) {
    for k in keys {
        let k = Key { frame: k.frame.max(0.0) + 0.0, ..k };
        match c.keys.binary_search_by(|o| o.frame.total_cmp(&k.frame)) {
            Ok(i) => c.keys[i] = k,
            Err(i) => c.keys.insert(i, k),
        }
    }
}

/// Take the selected keys out of the curve, returning them.
pub fn take(c: &mut AnimCurve, sel: &[f64]) -> Vec<Key> {
    let idx = picked(c, sel);
    let mut out = Vec::with_capacity(idx.len());
    for &i in idx.iter().rev() {
        out.push(c.keys.remove(i));
    }
    out.reverse();
    out
}

/// What a key slider does (Blender 4's Key › Blend menu, Houdini's
/// animation toolbar). The factor is -1..1 for the two-sided ones and
/// 0..1 for the rest (`Slide::two_sided`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slide {
    /// Toward the left neighbour's value (factor < 0) or the right's (> 0).
    Blend,
    /// Every key of the run onto one value between the neighbours: -1 the
    /// left one, 0 halfway, 1 the right one.
    Breakdown,
    /// Exaggerate (> 0) or flatten (< 0) the run's departure from the
    /// straight line between its neighbours.
    PushPull,
    /// The run onto an ease between its neighbours: < 0 eases in, > 0 out,
    /// 0 a straight line.
    Ease,
    /// Toward the channel's default value.
    Default,
    /// Gaussian smoothing, the factor the width (up to `SMOOTH_MAX` keys).
    Smooth,
    /// Remove keys — the factor is the fraction removed — keeping the ones
    /// that hold the shape.
    Decimate,
}

/// Widest smoothing kernel, in keys (sigma).
pub const SMOOTH_MAX: f32 = 8.0;

impl Slide {
    pub const ALL: [Slide; 7] = [
        Slide::Blend,
        Slide::Breakdown,
        Slide::PushPull,
        Slide::Ease,
        Slide::Default,
        Slide::Smooth,
        Slide::Decimate,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Slide::Blend => "blend to neighbour",
            Slide::Breakdown => "breakdown",
            Slide::PushPull => "push / pull",
            Slide::Ease => "ease",
            Slide::Default => "blend to default",
            Slide::Smooth => "smooth",
            Slide::Decimate => "decimate",
        }
    }
    pub fn two_sided(self) -> bool {
        matches!(self, Slide::Blend | Slide::Breakdown | Slide::PushPull | Slide::Ease)
    }
}

/// Run a slider op on the selected keys at `factor`. `default` is the
/// channel's default value (`None`: Default does nothing). Returns the
/// selected frames afterwards (Decimate removes some).
pub fn slide(c: &mut AnimCurve, sel: &[f64], kind: Slide, factor: f32, default: Option<f32>) -> Vec<f64> {
    let idx = picked(c, sel);
    match kind {
        Slide::Smooth => {
            smooth(c, &idx, factor.clamp(0.0, 1.0) * SMOOTH_MAX);
            return idx.iter().map(|&i| c.keys[i].frame).collect();
        }
        Slide::Decimate => return decimate(c, &idx, factor.clamp(0.0, 1.0)),
        Slide::Default => {
            if let Some(d) = default {
                let f = factor.clamp(0.0, 1.0);
                for &i in &idx {
                    let v = c.keys[i].value;
                    c.keys[i].value = v + (d - v) * f;
                }
            }
            return idx.iter().map(|&i| c.keys[i].frame).collect();
        }
        _ => {}
    }
    let f = factor.clamp(-1.0, 1.0);
    let n = c.keys.len();
    for (a, b) in runs(&idx) {
        // The unselected keys either side; a run at the curve's end leans
        // on its own end key.
        let l = c.keys[a.checked_sub(1).unwrap_or(a)];
        let r = c.keys[if b + 1 < n { b + 1 } else { b }];
        let span = (r.frame - l.frame).max(1e-9);
        for i in a..=b {
            let k = c.keys[i];
            let u = (((k.frame - l.frame) / span) as f32).clamp(0.0, 1.0);
            let line = l.value + (r.value - l.value) * u;
            c.keys[i].value = match kind {
                Slide::Blend if f < 0.0 => k.value + (l.value - k.value) * -f,
                Slide::Blend => k.value + (r.value - k.value) * f,
                Slide::Breakdown => l.value + (r.value - l.value) * (f + 1.0) * 0.5,
                Slide::PushPull => line + (k.value - line) * (1.0 + f),
                Slide::Ease => {
                    let p = 1.0 + 4.0 * f.abs();
                    let s = if f < 0.0 { u.powf(p) } else { 1.0 - (1.0 - u).powf(p) };
                    l.value + (r.value - l.value) * s
                }
                _ => k.value,
            };
        }
    }
    idx.iter().map(|&i| c.keys[i].frame).collect()
}

/// Gaussian smoothing of the selected keys' values, `sigma` in keys. The
/// unselected keys around them are read (so a run's ends blend into what
/// is either side) but never moved; the curve's own ends are extended
/// flat rather than pulled toward zero.
fn smooth(c: &mut AnimCurve, idx: &[usize], sigma: f32) {
    if sigma < 0.05 || idx.is_empty() {
        return;
    }
    let vals: Vec<f32> = c.keys.iter().map(|k| k.value).collect();
    let n = vals.len() as i64;
    let rad = (sigma * 3.0).ceil() as i64;
    let w: Vec<f32> = (-rad..=rad).map(|d| (-(d * d) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    for &i in idx {
        let (mut s, mut ws) = (0.0f32, 0.0f32);
        for d in -rad..=rad {
            let j = (i as i64 + d).clamp(0, n - 1) as usize;
            let wt = w[(d + rad) as usize];
            s += vals[j] * wt;
            ws += wt;
        }
        c.keys[i].value = s / ws;
    }
}

/// Remove `ratio` of the selected keys, cheapest first: the one whose
/// value is best predicted by the line between its neighbours goes, again
/// and again (Visvalingam's idea in one dimension). A run's first and
/// last keys always stay, so its ends — and what joins it to the rest of
/// the curve — do not move.
fn decimate(c: &mut AnimCurve, idx: &[usize], ratio: f32) -> Vec<f64> {
    let mut keep: Vec<f64> = idx.iter().map(|&i| c.keys[i].frame).collect();
    if ratio <= 0.0 {
        return keep;
    }
    let mut doomed: Vec<f64> = Vec::new();
    for (a, b) in runs(idx) {
        if b < a + 2 {
            continue;
        }
        // Working list of the run's keys (frame, value); the ends fixed.
        let mut pts: Vec<(f64, f32)> = (a..=b).map(|i| (c.keys[i].frame, c.keys[i].value)).collect();
        let removable = pts.len() - 2;
        let target = ((removable as f32) * ratio).round() as usize;
        for _ in 0..target {
            let mut best: Option<(usize, f32)> = None;
            for j in 1..pts.len() - 1 {
                let (p, q, r) = (pts[j - 1], pts[j], pts[j + 1]);
                let u = ((q.0 - p.0) / (r.0 - p.0).max(1e-9)) as f32;
                let err = (p.1 + (r.1 - p.1) * u - q.1).abs();
                if best.is_none_or(|(_, e)| err < e) {
                    best = Some((j, err));
                }
            }
            let Some((j, _)) = best else { break };
            doomed.push(pts.remove(j).0);
        }
    }
    c.keys.retain(|k| !doomed.contains(&k.frame));
    keep.retain(|f| !doomed.contains(f));
    keep
}

/// Undo wrap-around jumps (Blender's Euler filter, for one angle): where
/// two consecutive selected keys differ by more than half a `period`, every
/// later selected key shifts by whole periods so the angle turns the short
/// way. Degrees: `period` 360.
pub fn unwrap(c: &mut AnimCurve, sel: &[f64], period: f32) {
    let idx = picked(c, sel);
    let mut shift = 0.0f32;
    let mut prev: Option<f32> = None; // previous key's ORIGINAL value
    for &i in &idx {
        let v = c.keys[i].value;
        if let Some(p) = prev {
            let d = v - p;
            shift -= (d / period).round() * period;
        }
        prev = Some(v);
        c.keys[i].value = v + shift;
    }
}

/// Mirror the selected keys in TIME about `pivot`: the run plays
/// backwards. Handles swap sides, and each segment keeps its shape — the
/// interpolation of a segment lives on its LEFT key, which is the old
/// right key's left neighbour's, and an ease in becomes an ease out.
/// Returns the selected frames afterwards.
pub fn mirror_time(c: &mut AnimCurve, sel: &[f64], pivot: f64) -> Vec<f64> {
    let idx = picked(c, sel);
    if idx.is_empty() {
        return Vec::new();
    }
    // Interp of the segment ARRIVING at each picked key (from its left
    // neighbour), read before anything moves.
    let arriving: Vec<Option<Interp>> =
        idx.iter().map(|&i| i.checked_sub(1).map(|j| c.keys[j].interp)).collect();
    let left_picked: Vec<bool> = idx.iter().map(|&i| i > 0 && idx.contains(&(i - 1))).collect();
    let mut moved = take(c, sel);
    for (n, k) in moved.iter_mut().enumerate() {
        k.frame = 2.0 * pivot - k.frame;
        let (ti, to) = (k.tan_in, k.tan_out);
        k.tan_out = (-ti.0, ti.1);
        k.tan_in = (-to.0, to.1);
        std::mem::swap(&mut k.vec_in, &mut k.vec_out);
        // Only inside the mirrored run is "the segment arriving here" the
        // one that now leaves; a key whose left neighbour stayed put keeps
        // its own.
        if left_picked[n] {
            if let Some(m) = arriving[n] {
                k.interp = flip_ease(m);
            }
        }
    }
    let frames: Vec<f64> = moved.iter().map(|k| k.frame.max(0.0) + 0.0).collect();
    place(c, moved);
    frames
}

fn flip_ease(i: Interp) -> Interp {
    match i {
        Interp::Ease(e, EaseDir::In) => Interp::Ease(e, EaseDir::Out),
        Interp::Ease(e, EaseDir::Out) => Interp::Ease(e, EaseDir::In),
        o => o,
    }
}

/// Mirror the selected keys' VALUES about `pivot` (handles flip with them).
pub fn mirror_value(c: &mut AnimCurve, sel: &[f64], pivot: f32) {
    for i in picked(c, sel) {
        let k = &mut c.keys[i];
        k.value = 2.0 * pivot - k.value;
        k.tan_in.1 = -k.tan_in.1;
        k.tan_out.1 = -k.tan_out.1;
    }
}

/// Vector handles on the selected keys: bezier both sides, each handle
/// pointing at its neighbour.
pub fn vector_handles(c: &mut AnimCurve, sel: &[f64]) {
    for f in sel {
        c.ensure_bezier(*f);
        if let Some(i) = c.index_of(*f) {
            let k = &mut c.keys[i];
            k.broken = true;
            k.vec_in = true;
            k.vec_out = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(vals: &[(f64, f32)]) -> AnimCurve {
        let mut c = AnimCurve::default();
        for (f, v) in vals {
            c.set_key(*f, *v);
        }
        c
    }
    fn values(c: &AnimCurve) -> Vec<f32> {
        c.keys.iter().map(|k| k.value).collect()
    }

    #[test]
    fn blend_breakdown_and_push_pull_lean_on_the_neighbours() {
        let base = curve(&[(0.0, 0.0), (5.0, 3.0), (10.0, 10.0)]);
        let mut c = base.clone();
        slide(&mut c, &[5.0], Slide::Blend, 1.0, None);
        assert_eq!(values(&c), vec![0.0, 10.0, 10.0], "all the way to the right neighbour");
        let mut c = base.clone();
        slide(&mut c, &[5.0], Slide::Blend, -0.5, None);
        assert_eq!(values(&c)[1], 1.5, "half way to the left one");
        let mut c = base.clone();
        slide(&mut c, &[5.0], Slide::Breakdown, 0.0, None);
        assert_eq!(values(&c)[1], 5.0, "breakdown 0 is the middle of the neighbours");
        let mut c = base.clone();
        slide(&mut c, &[5.0], Slide::PushPull, -1.0, None);
        assert_eq!(values(&c)[1], 5.0, "pull all the way: onto the line");
        let mut c = base.clone();
        slide(&mut c, &[5.0], Slide::PushPull, 1.0, None);
        assert_eq!(values(&c)[1], 1.0, "push: twice as far off the line (5 - 2*2)");
        let mut c = base.clone();
        slide(&mut c, &[5.0], Slide::Ease, 0.0, None);
        assert_eq!(values(&c)[1], 5.0, "ease 0 is the straight line");
        let mut c = base;
        slide(&mut c, &[5.0], Slide::Default, 0.5, Some(1.0));
        assert_eq!(values(&c)[1], 2.0, "half way to the default");
    }

    #[test]
    fn smoothing_takes_jitter_out_and_leaves_unselected_keys() {
        let mut c = curve(&[]);
        for f in 0..40 {
            c.set_key(f as f64, f as f32 + if f % 2 == 0 { 1.0 } else { -1.0 });
        }
        let sel: Vec<f64> = (5..35).map(|f| f as f64).collect();
        let before = values(&c);
        slide(&mut c, &sel, Slide::Smooth, 0.5, None);
        let after = values(&c);
        assert_eq!(after[0..5], before[0..5], "unselected keys are only read");
        for f in 10..30 {
            assert!((after[f] - f as f32).abs() < 0.05, "frame {f}: {} left of the ±1 jitter", after[f]);
        }
    }

    #[test]
    fn decimate_keeps_the_corners_and_the_ends() {
        // A ramp up then down: the peak is the only key that matters.
        let mut c = curve(&[]);
        for f in 0..=20 {
            c.set_key(f as f64, if f <= 10 { f as f32 } else { 20.0 - f as f32 });
        }
        let sel: Vec<f64> = (0..=20).map(|f| f as f64).collect();
        let left = slide(&mut c, &sel, Slide::Decimate, 0.95, None);
        let frames: Vec<f64> = c.keys.iter().map(|k| k.frame).collect();
        assert_eq!(frames, vec![0.0, 10.0, 20.0], "everything but the shape-holding keys went");
        assert_eq!(left, frames);
        for t in [3.0, 10.0, 17.0] {
            let want = if t <= 10.0 { t } else { 20.0 - t };
            assert!((c.eval_at(t).unwrap() - want).abs() < 1e-4);
        }
        // Half: still the ends and the peak, and half the rest.
        let mut c = curve(&[]);
        for f in 0..=20 {
            c.set_key(f as f64, (f as f32 * 0.3).sin());
        }
        slide(&mut c, &sel, Slide::Decimate, 0.5, None);
        assert_eq!(c.keys.len(), 21 - 10);
        assert_eq!((c.keys[0].frame, c.keys.last().unwrap().frame), (0.0, 20.0));
    }

    #[test]
    fn unwrap_turns_the_short_way() {
        let mut c = curve(&[(0.0, 170.0), (1.0, -170.0), (2.0, -150.0), (3.0, 175.0)]);
        unwrap(&mut c, &[0.0, 1.0, 2.0, 3.0], 360.0);
        assert_eq!(values(&c), vec![170.0, 190.0, 210.0, 175.0 + 0.0], "{:?}", values(&c));
    }

    #[test]
    fn mirror_in_time_plays_the_run_backwards() {
        let mut c = curve(&[(0.0, 0.0), (10.0, 5.0), (30.0, 1.0)]);
        c.set_interp(0.0, Interp::Ease(crate::anim::Ease::Quad, EaseDir::In));
        let before: Vec<f32> = (0..=30).map(|t| c.eval_at(t as f32).unwrap()).collect();
        let sel = mirror_time(&mut c, &[0.0, 10.0, 30.0], 15.0);
        assert_eq!(sel, vec![30.0, 20.0, 0.0]);
        for t in 0..=30 {
            let want = before[30 - t];
            let got = c.eval_at(t as f32).unwrap();
            assert!((got - want).abs() < 1e-4, "frame {t}: {got} vs {want}");
        }
    }

    #[test]
    fn mirror_value_flips_about_the_pivot() {
        let mut c = curve(&[(0.0, 1.0), (4.0, 3.0)]);
        mirror_value(&mut c, &[0.0, 4.0], 2.0);
        assert_eq!(values(&c), vec![3.0, 1.0]);
    }

    #[test]
    fn vector_handles_point_at_the_neighbours_and_follow_them() {
        let mut c = curve(&[(0.0, 0.0), (9.0, 9.0), (18.0, 0.0)]);
        vector_handles(&mut c, &[9.0]);
        assert_eq!(c.in_handle(1), (-3.0, -3.0));
        assert_eq!(c.out_handle(1), (3.0, -3.0));
        // Move the neighbour: the handle follows.
        c.keys[2].value = 9.0;
        assert_eq!(c.out_handle(1), (3.0, 0.0));
    }

    #[test]
    fn place_replaces_a_key_on_the_same_frame() {
        let mut c = curve(&[(0.0, 0.0), (5.0, 1.0), (10.0, 2.0)]);
        let mut moved = take(&mut c, &[0.0]);
        moved[0].frame = 5.0;
        place(&mut c, moved);
        assert_eq!(c.keys.iter().map(|k| (k.frame, k.value)).collect::<Vec<_>>(), vec![(5.0, 0.0), (10.0, 2.0)]);
    }
}
