//! Right-angle noodles: straight runs that only ever go across or down (or
//! up), joined by rounded corners — never a diagonal. The graph's "right
//! angles" mode routes every wire through here instead of a bezier, and the
//! same polyline is what the slicer, dot cuts and drop-on-a-wire hit test,
//! so what is clicked is what is drawn.
//!
//! A route is decided by how it meets each end ([`End`]): a node's output
//! leaves DOWN and an input is entered from above, an expression cable runs
//! sideways, and a dot or the pointer has no direction of its own — so a dot
//! becomes the corner, the way dots are usually used.

use egui::{Pos2, Vec2};

/// How a route meets one of its ends.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum End {
    /// Travelling down through it: a node's output leaves downward, an input
    /// is entered from above.
    Down,
    /// Travelling sideways through it, along +x for a positive sign and −x
    /// for a negative one: an expression cable's ends.
    Side(f32),
    /// No direction of its own: a dot, the pointer.
    Free,
}

/// Sizes, in screen points.
#[derive(Clone, Copy, Debug)]
pub struct Style {
    /// Corner radius, where the runs leave room for it.
    pub radius: f32,
    /// The shortest run out of a socket before a route may turn back.
    pub stub: f32,
    /// How far a route that has to double back swings wide of its ends.
    pub clear: f32,
    /// The spacing between parallel runs that [`lanes`] pulls apart.
    pub lane: f32,
}

impl Style {
    /// The graph's sizes at `zoom`: proportional to the nodes, so a zoomed
    /// out graph keeps its shape.
    pub fn at(zoom: f32) -> Self {
        Self { radius: 14.0 * zoom, stub: 14.0 * zoom, clear: 48.0 * zoom, lane: 8.0 * zoom }
    }
}

/// Ends closer than this across the run are drawn as one straight line: a
/// jog of a pixel reads as a kink, not as a corner.
const SNAP: f32 = 1.5;

/// The route from `a` to `b`, corners rounded.
pub fn route(a: Pos2, ea: End, b: Pos2, eb: End, st: &Style) -> Vec<Pos2> {
    round(&corners(a, ea, b, eb, st), st.radius)
}

/// The route's corner points: a polyline whose every run is horizontal or
/// vertical (one that SNAP made straight may be off by under 1.5 points).
pub fn corners(a: Pos2, ea: End, b: Pos2, eb: End, st: &Style) -> Vec<Pos2> {
    let p = Pos2::new;
    let pts = match (ea, eb) {
        (End::Side(sa), End::Side(_)) => {
            // Mirror a leftward cable into a rightward one and back.
            let s = if sa < 0.0 { -1.0 } else { 1.0 };
            let f = |q: Pos2| p(q.x * s, q.y);
            let (a, b) = (f(a), f(b));
            let pts = if (a.y - b.y).abs() < SNAP && b.x >= a.x {
                vec![a, b]
            } else if b.x - a.x >= 2.0 * st.stub {
                let m = (a.x + b.x) * 0.5;
                vec![a, p(m, a.y), p(m, b.y), b]
            } else {
                let (x0, x1) = (a.x + st.stub, b.x - st.stub);
                let lane = if (a.y - b.y).abs() >= 2.0 * st.clear {
                    (a.y + b.y) * 0.5
                } else {
                    a.y.max(b.y) + st.clear
                };
                vec![a, p(x0, a.y), p(x0, lane), p(x1, lane), p(x1, b.y), b]
            };
            pts.into_iter().map(f).collect()
        }
        (End::Side(s), End::Free) => {
            let s = if s < 0.0 { -1.0 } else { 1.0 };
            if (a.y - b.y).abs() < SNAP && s * (b.x - a.x) >= 0.0 {
                vec![a, b]
            } else if s * (b.x - a.x) >= st.stub * 0.5 {
                vec![a, p(b.x, a.y), b]
            } else {
                let x0 = a.x + s * st.stub;
                vec![a, p(x0, a.y), p(x0, b.y), b]
            }
        }
        (End::Free, End::Side(s)) => {
            let s = if s < 0.0 { -1.0 } else { 1.0 };
            if (a.y - b.y).abs() < SNAP && s * (b.x - a.x) >= 0.0 {
                vec![a, b]
            } else if s * (b.x - a.x) >= st.stub * 0.5 {
                vec![a, p(a.x, b.y), b]
            } else {
                let x1 = b.x - s * st.stub;
                vec![a, p(x1, a.y), p(x1, b.y), b]
            }
        }
        (End::Down, End::Free) => {
            if (a.x - b.x).abs() < SNAP {
                vec![a, b]
            } else if b.y >= a.y + st.stub * 0.5 {
                // Down to the dot's height, then across into it.
                vec![a, p(a.x, b.y), b]
            } else {
                // Above the socket: down a stub first, then across and up.
                let y0 = a.y + st.stub;
                vec![a, p(a.x, y0), p(b.x, y0), b]
            }
        }
        (End::Free, End::Down) => {
            if (a.x - b.x).abs() < SNAP {
                vec![a, b]
            } else if b.y >= a.y + st.stub * 0.5 {
                // Across to the input's line, then down into it.
                vec![a, p(b.x, a.y), b]
            } else {
                let y1 = b.y - st.stub;
                vec![a, p(a.x, y1), p(b.x, y1), b]
            }
        }
        (End::Free, End::Free) => {
            if (a.x - b.x).abs() < SNAP || (a.y - b.y).abs() < SNAP {
                vec![a, b]
            } else {
                vec![a, p(a.x, b.y), b]
            }
        }
        (End::Down, End::Side(s)) => {
            // Into a node's side socket: down to its height and in sideways,
            // or round a short step when it sits above the output.
            let s = if s < 0.0 { -1.0 } else { 1.0 };
            if b.y >= a.y - SNAP && s * (b.x - a.x) >= st.stub * 0.5 {
                vec![a, p(a.x, b.y), b]
            } else {
                let (x1, y0) = (b.x - s * st.stub, a.y + st.stub);
                vec![a, p(a.x, y0), p(x1, y0), p(x1, b.y), b]
            }
        }
        (End::Side(_), End::Down) => vec![a, p(b.x, a.y), b],
        (End::Down, End::Down) => {
            if (a.x - b.x).abs() < SNAP && b.y >= a.y {
                vec![a, b]
            } else if b.y - a.y >= 2.0 * st.stub {
                // The plain case: down, across halfway, down.
                let m = (a.y + b.y) * 0.5;
                vec![a, p(a.x, m), p(b.x, m), b]
            } else {
                // The input is level with or above the output: hook out
                // under the source, run up beside, drop in over the target.
                let (y0, y1) = (a.y + st.stub, b.y - st.stub);
                let lane = if (a.x - b.x).abs() >= 2.0 * st.clear {
                    (a.x + b.x) * 0.5
                } else {
                    a.x.max(b.x) + st.clear
                };
                vec![a, p(a.x, y0), p(lane, y0), p(lane, y1), p(b.x, y1), b]
            }
        }
    };
    simplify(pts)
}

/// Drop repeated points and the middle of any straight run.
fn simplify(pts: Vec<Pos2>) -> Vec<Pos2> {
    let mut out: Vec<Pos2> = Vec::with_capacity(pts.len());
    for q in pts {
        if out.last().is_some_and(|l| l.distance(q) < 1e-3) {
            continue;
        }
        if out.len() >= 2 {
            let (a, b) = (out[out.len() - 2], out[out.len() - 1]);
            let (u, w) = (b - a, q - b);
            if u.x * w.y - u.y * w.x == 0.0 && u.dot(w) > 0.0 {
                out.pop();
            }
        }
        out.push(q);
    }
    out
}

/// Round each corner with a quarter arc of up to `radius`. A corner takes
/// at most half of a run it shares with the next corner, and nearly all of
/// a run that ends at a socket — so runs never overlap, and a wire still
/// arrives at a socket straight.
pub fn round(pts: &[Pos2], radius: f32) -> Vec<Pos2> {
    let n = pts.len();
    if n < 3 || radius <= 0.0 {
        return pts.to_vec();
    }
    let budget = |i: usize| -> f32 {
        // Run i joins pts[i] and pts[i + 1].
        let len = pts[i].distance(pts[i + 1]);
        if i == 0 || i + 2 == n {
            len * 0.9
        } else {
            len * 0.5
        }
    };
    let mut out = vec![pts[0]];
    for i in 1..n - 1 {
        let (p, v, q) = (pts[i - 1], pts[i], pts[i + 1]);
        let (u, w) = ((v - p).normalized(), (q - v).normalized());
        let r = radius.min(budget(i - 1)).min(budget(i));
        // Only square turns get an arc; a doubling-back keeps its point.
        if r < 0.5 || u.dot(w).abs() > 0.01 {
            out.push(v);
            continue;
        }
        let c = v - u * r + w * r;
        let steps = ((r / 2.0).ceil() as usize).clamp(3, 10);
        for k in 0..=steps {
            let t = k as f32 / steps as f32 * std::f32::consts::FRAC_PI_2;
            out.push(c - w * (r * t.cos()) + u * (r * t.sin()));
        }
    }
    out.push(pts[n - 1]);
    out
}

fn seg_dist(q: Pos2, a: Pos2, b: Pos2) -> (f32, Pos2) {
    let ab = b - a;
    let len2 = ab.length_sq();
    let t = if len2 <= 1e-9 { 0.0 } else { ((q - a).dot(ab) / len2).clamp(0.0, 1.0) };
    let at = a + ab * t;
    (at.distance(q), at)
}

/// The nearest distance from `q` to the path.
pub fn min_dist(pts: &[Pos2], q: Pos2) -> f32 {
    match pts {
        [] => f32::MAX,
        [p] => p.distance(q),
        _ => pts.windows(2).map(|s| seg_dist(q, s[0], s[1]).0).fold(f32::MAX, f32::min),
    }
}

/// The point on the path nearest `q`.
pub fn nearest(pts: &[Pos2], q: Pos2) -> Pos2 {
    match pts {
        [] => q,
        [p] => *p,
        _ => pts
            .windows(2)
            .map(|s| seg_dist(q, s[0], s[1]))
            .min_by(|x, y| x.0.total_cmp(&y.0))
            .map(|(_, at)| at)
            .unwrap_or(q),
    }
}

fn segs_cross(a: Pos2, b: Pos2, c: Pos2, d: Pos2) -> bool {
    let cross = |o: Pos2, p: Pos2, q: Pos2| (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    let (d1, d2) = (cross(c, d, a), cross(c, d, b));
    let (d3, d4) = (cross(a, b, c), cross(a, b, d));
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

/// Does the segment `a`–`b` cross the path? (The slicer's question.)
pub fn hits_segment(pts: &[Pos2], a: Pos2, b: Pos2) -> bool {
    pts.windows(2).any(|s| segs_cross(a, b, s[0], s[1]))
}

/// The point `u` of the way along the path, by length.
pub fn point_at(pts: &[Pos2], u: f32) -> Pos2 {
    let total: f32 = pts.windows(2).map(|s| s[0].distance(s[1])).sum();
    let mut left = total * u.clamp(0.0, 1.0);
    for s in pts.windows(2) {
        let l = s[0].distance(s[1]);
        if left <= l && l > 0.0 {
            return s[0] + (s[1] - s[0]) * (left / l);
        }
        left -= l;
    }
    pts.last().copied().unwrap_or(Pos2::ZERO)
}

/// `n + 1` points evenly spaced along the path, ends included: for dashes
/// and colour ramps, which want steps of equal length.
pub fn resample(pts: &[Pos2], n: usize) -> Vec<Pos2> {
    let n = n.max(1);
    let total: f32 = pts.windows(2).map(|s| s[0].distance(s[1])).sum();
    if pts.len() < 2 || total <= 0.0 {
        return vec![pts.first().copied().unwrap_or(Pos2::ZERO); n + 1];
    }
    let mut out = Vec::with_capacity(n + 1);
    let (mut i, mut walked) = (0usize, 0.0f32);
    for k in 0..=n {
        let want = total * k as f32 / n as f32;
        while i + 2 < pts.len() && walked + pts[i].distance(pts[i + 1]) < want {
            walked += pts[i].distance(pts[i + 1]);
            i += 1;
        }
        let l = pts[i].distance(pts[i + 1]);
        let t = if l > 0.0 { ((want - walked) / l).clamp(0.0, 1.0) } else { 0.0 };
        out.push(pts[i] + (pts[i + 1] - pts[i]) * t);
    }
    out
}

/// The path cut into `n` pieces of equal length, each keeping every corner
/// point it spans — so a ramp or a dash drawn piece by piece follows the
/// route exactly, where chords between resampled points would cut a corner
/// on a long cable.
pub fn split(pts: &[Pos2], n: usize) -> Vec<Vec<Pos2>> {
    let n = n.max(1);
    let marks = resample(pts, n);
    if pts.len() < 2 {
        return vec![marks.clone(); n];
    }
    let total: f32 = pts.windows(2).map(|s| s[0].distance(s[1])).sum();
    let mut out = Vec::with_capacity(n);
    let (mut i, mut walked) = (0usize, 0.0f32);
    for k in 0..n {
        let end = total * (k + 1) as f32 / n as f32;
        let mut piece = vec![marks[k]];
        // Every vertex strictly inside this piece.
        while i + 1 < pts.len() && walked + pts[i].distance(pts[i + 1]) < end {
            walked += pts[i].distance(pts[i + 1]);
            i += 1;
            if piece.last().is_none_or(|l| l.distance(pts[i]) > 1e-4) {
                piece.push(pts[i]);
            }
        }
        if piece.last().is_none_or(|l| l.distance(marks[k + 1]) > 1e-4) || piece.len() == 1 {
            piece.push(marks[k + 1]);
        }
        out.push(piece);
    }
    out
}

/// The path moved `d` points to the left of its travel (right for a
/// negative `d`): one strand of a fat cable. Corners are mitred, so the
/// copy stays parallel through a square turn.
pub fn offset(pts: &[Pos2], d: f32) -> Vec<Pos2> {
    let n = pts.len();
    if n < 2 {
        return pts.to_vec();
    }
    let normal = |a: Pos2, b: Pos2| -> Option<Vec2> {
        let t = b - a;
        (t.length() > 1e-6).then(|| Vec2::new(t.y, -t.x).normalized())
    };
    (0..n)
        .map(|i| {
            let before = (i > 0).then(|| normal(pts[i - 1], pts[i])).flatten();
            let after = (i + 1 < n).then(|| normal(pts[i], pts[i + 1])).flatten();
            let m = match (before, after) {
                (Some(a), Some(b)) => {
                    let m = a + b;
                    if m.length() < 1e-3 {
                        a
                    } else {
                        let m = m.normalized();
                        m / m.dot(a).max(0.3)
                    }
                }
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => Vec2::ZERO,
            };
            pts[i] + m * d
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Lanes: parallel runs pulled apart, like lines on a subway map
// ---------------------------------------------------------------------------

/// One wire for [`lanes`]: its ends, and the group it belongs to. Wires of
/// one group (one output feeding several inputs) may share a run, the way
/// a line branches; wires of different groups never do.
#[derive(Clone, Copy, Debug)]
pub struct LaneWire {
    pub group: u64,
    pub a: Pos2,
    pub ea: End,
    pub b: Pos2,
    pub eb: End,
}

/// A run between two corners, as `lanes` sees it.
struct Run {
    wire: usize,
    /// Index of the run's first point in the wire's corner list.
    seg: usize,
    /// Where it lies across its axis (y for a horizontal run).
    at: f32,
    lo: f32,
    hi: f32,
    /// Each end: where along the run it is, and which way the neighbouring
    /// run leaves (+1 toward larger `at`).
    ends: [(f32, f32); 2],
    /// How far `at` may move each way before a neighbouring run flips.
    room_neg: f32,
    room_pos: f32,
}

/// Every wire's corner points, with runs that would lie on top of another
/// group's run spread into parallel lanes `st.lane` apart. Only runs
/// between two corners move (the ends stay on their sockets), never so far
/// that a neighbouring run turns back on itself, and the lanes of a bundle
/// are ordered so that as few wires as possible cross on their way in and
/// out. Horizontal runs are spread first, then vertical ones.
pub fn lanes(wires: &[LaneWire], st: &Style) -> Vec<Vec<Pos2>> {
    let mut pts: Vec<Vec<Pos2>> =
        wires.iter().map(|w| corners(w.a, w.ea, w.b, w.eb, st)).collect();
    if st.lane > 0.0 {
        for horizontal in [true, false] {
            spread(&mut pts, wires, st, horizontal);
        }
    }
    pts
}

/// `lanes`, rounded: the drawn routes.
pub fn lane_routes(wires: &[LaneWire], st: &Style) -> Vec<Vec<Pos2>> {
    lanes(wires, st).iter().map(|c| round(c, st.radius)).collect()
}

fn spread(pts: &mut [Vec<Pos2>], wires: &[LaneWire], st: &Style, horizontal: bool) {
    // (across, along) for a point, in the axis being spread.
    let ax = |p: Pos2| if horizontal { (p.y, p.x) } else { (p.x, p.y) };
    let mut runs: Vec<Run> = Vec::new();
    for (w, c) in pts.iter().enumerate() {
        let n = c.len();
        if n < 4 {
            continue;
        }
        for j in 1..n - 2 {
            let ((a0, l0), (a1, l1)) = (ax(c[j]), ax(c[j + 1]));
            if (a0 - a1).abs() > 0.5 || (l0 - l1).abs() < 0.5 {
                continue; // not a run along this axis
            }
            let (pa, _) = ax(c[j - 1]);
            let (na, _) = ax(c[j + 2]);
            let (dp, dn) = (pa - a0, na - a1);
            let mut room = [f32::MAX, f32::MAX]; // [neg, pos]
            for d in [dp, dn] {
                // Keep a bit of every neighbour, so its corner still rounds.
                let keep = (st.radius * 0.6).min(d.abs() * 0.5);
                let r = (d.abs() - keep).max(0.0);
                let k = if d < 0.0 { 0 } else { 1 };
                room[k] = room[k].min(r);
            }
            runs.push(Run {
                wire: w,
                seg: j,
                at: a0,
                lo: l0.min(l1),
                hi: l0.max(l1),
                ends: [(l0, dp.signum()), (l1, dn.signum())],
                room_neg: room[0],
                room_pos: room[1],
            });
        }
    }
    if runs.len() < 2 {
        return;
    }
    // Bundles: runs closer than a lane across, that overlap along.
    runs.sort_by(|a, b| a.at.total_cmp(&b.at));
    let mut parent: Vec<usize> = (0..runs.len()).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut i = i;
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for i in 0..runs.len() {
        for j in i + 1..runs.len() {
            if runs[j].at - runs[i].at >= st.lane {
                break;
            }
            let overlap = runs[i].hi.min(runs[j].hi) - runs[i].lo.max(runs[j].lo);
            if overlap > -st.lane * 0.5 {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                parent[a] = b;
            }
        }
    }
    let mut bundles: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for i in 0..runs.len() {
        let r = find(&mut parent, i);
        bundles.entry(r).or_default().push(i);
    }
    for members in bundles.values() {
        // Tracks: one per group in the bundle.
        let mut tracks: Vec<(u64, Vec<usize>)> = Vec::new();
        for &m in members {
            let g = wires[runs[m].wire].group;
            match tracks.iter_mut().find(|t| t.0 == g) {
                Some(t) => t.1.push(m),
                None => tracks.push((g, vec![m])),
            }
        }
        let k = tracks.len();
        if k < 2 {
            continue;
        }
        let order = order_tracks(&runs, &tracks);
        let centre = members.iter().map(|m| runs[*m].at).sum::<f32>() / members.len() as f32;
        // The widest spacing up to a lane that every run has room for.
        let want = |pitch: f32| -> Vec<f32> {
            order
                .iter()
                .enumerate()
                .map(|(rank, _)| centre + (rank as f32 - (k as f32 - 1.0) * 0.5) * pitch)
                .collect()
        };
        let fits = |targets: &[f32]| {
            order.iter().zip(targets).all(|(t, at)| {
                tracks[*t].1.iter().all(|m| {
                    let d = at - runs[*m].at;
                    d >= -runs[*m].room_neg - 1e-3 && d <= runs[*m].room_pos + 1e-3
                })
            })
        };
        let mut targets = None;
        for step in 0..8 {
            let t = want(st.lane * (1.0 - step as f32 / 8.0));
            if fits(&t) {
                targets = Some(t);
                break;
            }
        }
        // Nothing fits as a whole: each run goes as near its lane as it can.
        let targets = targets.unwrap_or_else(|| want(st.lane));
        for (t, at) in order.iter().zip(targets) {
            for &m in &tracks[*t].1 {
                let r = &runs[m];
                let at = at.clamp(r.at - r.room_neg, r.at + r.room_pos);
                let c = &mut pts[r.wire];
                for q in [r.seg, r.seg + 1] {
                    if horizontal {
                        c[q].y = at;
                    } else {
                        c[q].x = at;
                    }
                }
            }
        }
    }
}

/// The order of a bundle's tracks across it (first = smallest `at`) that
/// crosses the fewest wires: an end that leaves toward smaller `at` crosses
/// every track on that side whose run spans it, and likewise the other way.
fn order_tracks(runs: &[Run], tracks: &[(u64, Vec<usize>)]) -> Vec<usize> {
    let k = tracks.len();
    // cost[i][j]: crossings when track i lies on the smaller side of j.
    let spans = |t: usize, x: f32| {
        tracks[t].1.iter().any(|m| x > runs[*m].lo + 0.5 && x < runs[*m].hi - 0.5)
    };
    let mut cost = vec![vec![0u32; k]; k];
    for i in 0..k {
        for j in 0..k {
            if i == j {
                continue;
            }
            let mut c = 0;
            for &m in &tracks[j].1 {
                for (x, dir) in runs[m].ends {
                    if dir < 0.0 && spans(i, x) {
                        c += 1;
                    }
                }
            }
            for &m in &tracks[i].1 {
                for (x, dir) in runs[m].ends {
                    if dir > 0.0 && spans(j, x) {
                        c += 1;
                    }
                }
            }
            cost[i][j] = c;
        }
    }
    let total = |o: &[usize]| -> u32 {
        let mut s = 0;
        for a in 0..o.len() {
            for b in a + 1..o.len() {
                s += cost[o[a]][o[b]];
            }
        }
        s
    };
    // Start from where the runs already lie, then improve.
    let mut order: Vec<usize> = (0..k).collect();
    let mean = |t: usize| {
        tracks[t].1.iter().map(|m| runs[*m].at).sum::<f32>() / tracks[t].1.len() as f32
    };
    order.sort_by(|a, b| mean(*a).total_cmp(&mean(*b)).then(tracks[*a].0.cmp(&tracks[*b].0)));
    if k <= 6 {
        let mut best = (total(&order), order.clone());
        permute(&mut order.clone(), 0, &mut |o| {
            let c = total(o);
            if c < best.0 {
                best = (c, o.to_vec());
            }
        });
        return best.1;
    }
    // Too many to try them all: swap neighbours while it helps.
    let mut improved = true;
    while improved {
        improved = false;
        for a in 0..k - 1 {
            if cost[order[a + 1]][order[a]] < cost[order[a]][order[a + 1]] {
                order.swap(a, a + 1);
                improved = true;
            }
        }
    }
    order
}

fn permute(o: &mut Vec<usize>, i: usize, f: &mut impl FnMut(&[usize])) {
    if i == o.len() {
        f(o);
        return;
    }
    for j in i..o.len() {
        o.swap(i, j);
        permute(o, i + 1, f);
        o.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st() -> Style {
        Style::at(1.0)
    }

    /// Every straight run is horizontal or vertical: the point of the mode.
    fn square(pts: &[Pos2]) -> bool {
        pts.windows(2).all(|s| (s[0].x - s[1].x).abs() < 1e-3 || (s[0].y - s[1].y).abs() < 1e-3)
    }

    #[test]
    fn a_wire_down_and_across_is_a_z_with_square_runs() {
        let (a, b) = (Pos2::new(0.0, 0.0), Pos2::new(120.0, 200.0));
        let c = corners(a, End::Down, b, End::Down, &st());
        assert_eq!(c, vec![a, Pos2::new(0.0, 100.0), Pos2::new(120.0, 100.0), b]);
        assert!(square(&c));
        let r = route(a, End::Down, b, End::Down, &st());
        assert_eq!((r[0], *r.last().unwrap()), (a, b), "ends land on the sockets");
        // Rounded corners: no point strays outside the corner box.
        for q in &r {
            assert!(q.x >= -1e-3 && q.x <= 120.001 && q.y >= -1e-3 && q.y <= 200.001, "{q:?}");
        }
        // It still leaves straight down and arrives straight down.
        assert!((r[1].x - a.x).abs() < 1e-3 && r[1].y > a.y);
        let k = r.len();
        assert!((r[k - 2].x - b.x).abs() < 1e-3 && r[k - 2].y < b.y);
    }

    #[test]
    fn a_straight_drop_has_no_corners() {
        let (a, b) = (Pos2::new(50.0, 0.0), Pos2::new(50.8, 90.0));
        assert_eq!(route(a, End::Down, b, End::Down, &st()), vec![a, b]);
    }

    #[test]
    fn an_input_above_its_source_hooks_round_instead_of_cutting_up() {
        // Target level with / above the source: down, beside, up, over, in.
        let (a, b) = (Pos2::new(0.0, 100.0), Pos2::new(10.0, 40.0));
        let c = corners(a, End::Down, b, End::Down, &st());
        assert!(square(&c));
        assert!(c[1].y > a.y, "leaves downward first: {c:?}");
        assert!(c[c.len() - 2].y < b.y, "drops into the input from above: {c:?}");
        // Close together sideways: the run swings wide of both.
        assert!(c.iter().any(|q| q.x >= 10.0 + 48.0 - 1e-3), "{c:?}");
        // Far apart sideways: it runs between them.
        let c = corners(a, End::Down, Pos2::new(300.0, 40.0), End::Down, &st());
        assert!(c.iter().any(|q| (q.x - 150.0).abs() < 1e-3), "{c:?}");
    }

    #[test]
    fn a_dot_is_the_corner() {
        let st = st();
        // Output down to a dot off to the side: one corner, under the source.
        let c = corners(Pos2::new(0.0, 0.0), End::Down, Pos2::new(80.0, 60.0), End::Free, &st);
        assert_eq!(c, vec![Pos2::new(0.0, 0.0), Pos2::new(0.0, 60.0), Pos2::new(80.0, 60.0)]);
        // Dot across to an input below: across first, then down into it.
        let c = corners(Pos2::new(80.0, 60.0), End::Free, Pos2::new(200.0, 160.0), End::Down, &st);
        assert_eq!(c, vec![Pos2::new(80.0, 60.0), Pos2::new(200.0, 60.0), Pos2::new(200.0, 160.0)]);
        // Two dots level with each other: one straight run.
        let c = corners(Pos2::new(0.0, 60.0), End::Free, Pos2::new(90.0, 60.0), End::Free, &st);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn a_side_socket_is_entered_sideways() {
        let st = st();
        // A matte off to the left, level with the socket: one straight run in.
        let (a, b) = (Pos2::new(0.0, 100.0), Pos2::new(200.0, 102.0));
        assert_eq!(corners(a, End::Down, b, End::Side(1.0), &st), vec![a, Pos2::new(0.0, 102.0), b]);
        // Socket above the output: down a step, across, up, and in from the side.
        let b = Pos2::new(200.0, 40.0);
        let c = corners(a, End::Down, b, End::Side(1.0), &st);
        assert!(square(&c) && c[1].y > a.y, "{c:?}");
        let k = c.len();
        assert!((c[k - 2].y - b.y).abs() < 1e-3 && c[k - 2].x < b.x, "arrives travelling +x: {c:?}");
        // A socket facing right is entered travelling −x.
        let c = corners(Pos2::new(300.0, 0.0), End::Down, Pos2::new(100.0, 80.0), End::Side(-1.0), &st);
        let k = c.len();
        assert!((c[k - 2].y - 80.0).abs() < 1e-3 && c[k - 2].x > 100.0, "{c:?}");
    }

    #[test]
    fn a_cable_runs_sideways_both_ways() {
        let st = st();
        let (a, b) = (Pos2::new(0.0, 0.0), Pos2::new(200.0, 80.0));
        let c = corners(a, End::Side(1.0), b, End::Side(1.0), &st);
        assert_eq!(c, vec![a, Pos2::new(100.0, 0.0), Pos2::new(100.0, 80.0), b]);
        // Leftward: the mirror image, leaving and arriving travelling −x.
        let (a, b) = (Pos2::new(200.0, 0.0), Pos2::new(0.0, 80.0));
        let c = corners(a, End::Side(-1.0), b, End::Side(-1.0), &st);
        assert_eq!(c, vec![a, Pos2::new(100.0, 0.0), Pos2::new(100.0, 80.0), b]);
        // Doubling back stays square.
        let c = corners(Pos2::new(100.0, 0.0), End::Side(1.0), Pos2::new(40.0, 30.0), End::Side(1.0), &st);
        assert!(square(&c) && c[1].x > 100.0 && c[c.len() - 2].x < 40.0, "{c:?}");
    }

    #[test]
    fn corners_shrink_to_fit_short_runs() {
        // A 10-point jog cannot take a 14-point radius.
        let pts = [Pos2::new(0.0, 0.0), Pos2::new(0.0, 50.0), Pos2::new(10.0, 50.0), Pos2::new(10.0, 100.0)];
        let r = round(&pts, 14.0);
        for q in &r {
            assert!(q.x >= -1e-3 && q.x <= 10.001, "{q:?}");
        }
        // Monotone down: no run doubles back on itself.
        assert!(r.windows(2).all(|s| s[1].y >= s[0].y - 1e-3));
    }

    #[test]
    fn hits_and_nearest_follow_the_drawn_path() {
        let r = route(Pos2::new(0.0, 0.0), End::Down, Pos2::new(120.0, 200.0), End::Down, &st());
        // On the horizontal run, not on the straight line between the ends.
        assert!(min_dist(&r, Pos2::new(60.0, 100.0)) < 0.5);
        assert!(min_dist(&r, Pos2::new(60.0, 100.0 * 200.0 / 120.0 * 0.5)) > 5.0);
        assert!(nearest(&r, Pos2::new(60.0, 110.0)).distance(Pos2::new(60.0, 100.0)) < 0.5);
        // A slice across the vertical run cuts it; one beside the path does not.
        assert!(hits_segment(&r, Pos2::new(-10.0, 40.0), Pos2::new(10.0, 40.0)));
        assert!(!hits_segment(&r, Pos2::new(20.0, 20.0), Pos2::new(40.0, 20.0)));
    }

    #[test]
    fn resampling_and_offsets_keep_the_shape() {
        let r = route(Pos2::new(0.0, 0.0), End::Down, Pos2::new(120.0, 200.0), End::Down, &st());
        let s = resample(&r, 20);
        assert_eq!(s.len(), 21);
        assert_eq!((s[0], s[20]), (r[0], *r.last().unwrap()));
        let gaps: Vec<f32> = s.windows(2).map(|w| w[0].distance(w[1])).collect();
        let (lo, hi) = gaps.iter().fold((f32::MAX, 0.0f32), |(l, h), g| (l.min(*g), h.max(*g)));
        assert!(hi - lo < 2.0, "even steps (chords round a corner run short): {lo}..{hi}");
        assert!(point_at(&r, 0.5).distance(Pos2::new(60.0, 100.0)) < 1.0);
        // A strand 3 points over runs parallel: 3 points off every run.
        let o = offset(&corners(Pos2::new(0.0, 0.0), End::Down, Pos2::new(120.0, 200.0), End::Down, &st()), 3.0);
        assert_eq!(o[0], Pos2::new(3.0, 0.0), "left of a downward run is +x on screen");
        assert!((o[1].x - 3.0).abs() < 1e-3 && (o[1].y - 97.0).abs() < 1e-3, "mitred corner {:?}", o[1]);
        assert!(square(&o));
    }

    #[test]
    fn pieces_keep_the_corners_a_long_cable_would_cut() {
        // Long runs, one rounded corner: 4 pieces of ~250 points each.
        let r = route(Pos2::new(0.0, 0.0), End::Free, Pos2::new(500.0, 500.0), End::Free, &st());
        let pieces = split(&r, 4);
        assert_eq!(pieces.len(), 4);
        assert_eq!(pieces[0][0], r[0]);
        assert_eq!(*pieces[3].last().unwrap(), *r.last().unwrap());
        for w in pieces.windows(2) {
            assert!(w[0].last().unwrap().distance(w[1][0]) < 1e-3, "pieces join end to end");
        }
        // No straight step is diagonal and longer than a corner's chord.
        for piece in &pieces {
            for s in piece.windows(2) {
                let d = s[1] - s[0];
                let diagonal = d.x.abs() > 1e-3 && d.y.abs() > 1e-3;
                assert!(!diagonal || d.length() < 14.0, "a diagonal cut {:?}", s);
            }
        }
    }
    /// Crossings between two square polylines (proper crossings only).
    fn crossings(a: &[Pos2], b: &[Pos2]) -> usize {
        let mut n = 0;
        for s in a.windows(2) {
            for t in b.windows(2) {
                if segs_cross(s[0], s[1], t[0], t[1]) {
                    n += 1;
                }
            }
        }
        n
    }

    fn lw(group: u64, a: (f32, f32), b: (f32, f32)) -> LaneWire {
        LaneWire { group, a: Pos2::new(a.0, a.1), ea: End::Down, b: Pos2::new(b.0, b.1), eb: End::Down }
    }

    #[test]
    fn runs_on_one_line_split_into_lanes_without_crossing() {
        // Three sources on one row into three inputs of one node below
        // (a Layer's bar): their across-runs all sit halfway, on one line.
        let st = st();
        let wires = [
            lw(1, (0.0, 0.0), (300.0, 200.0)),
            lw(2, (100.0, 0.0), (310.0, 200.0)),
            lw(3, (200.0, 0.0), (320.0, 200.0)),
        ];
        let plain: Vec<Vec<Pos2>> =
            wires.iter().map(|w| corners(w.a, w.ea, w.b, w.eb, &st)).collect();
        assert!(plain.iter().all(|c| (c[1].y - 100.0).abs() < 1e-3), "{plain:?}");
        let laned = lanes(&wires, &st);
        let mut ys: Vec<f32> = laned.iter().map(|c| c[1].y).collect();
        for c in &laned {
            assert!(square(c), "{c:?}");
        }
        ys.sort_by(f32::total_cmp);
        assert!(ys[1] - ys[0] >= st.lane - 1e-3 && ys[2] - ys[1] >= st.lane - 1e-3, "{ys:?}");
        for i in 0..3 {
            for j in i + 1..3 {
                assert_eq!(crossings(&laned[i], &laned[j]), 0, "{i} x {j}: {laned:?}");
            }
        }
        // Ends never move.
        for (w, c) in wires.iter().zip(&laned) {
            assert_eq!((c[0], *c.last().unwrap()), (w.a, w.b));
        }
    }

    #[test]
    fn one_output_feeding_two_inputs_shares_its_run() {
        let st = st();
        let wires = [lw(7, (0.0, 0.0), (200.0, 200.0)), lw(7, (0.0, 0.0), (220.0, 200.0))];
        let laned = lanes(&wires, &st);
        assert_eq!(laned[0][1].y, laned[1][1].y, "a branching line, not two");
    }

    #[test]
    fn lanes_never_flip_a_short_drop() {
        // Rows close together: the drops are short, so the lanes squeeze
        // rather than push a run past its socket.
        let st = st();
        let wires = [lw(1, (0.0, 0.0), (300.0, 40.0)), lw(2, (100.0, 0.0), (310.0, 40.0))];
        for c in lanes(&wires, &st) {
            assert!(c.windows(2).all(|s| s[1].y >= s[0].y - 1e-3), "turned back: {c:?}");
        }
    }
}
