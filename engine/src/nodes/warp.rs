//! Warp: pairs of curves, each a SOURCE curve and a DESTINATION curve with
//! the same points. Whatever lay along a source curve is moved onto its
//! destination curve, point for point, and the picture between and around
//! the curves follows smoothly, fading back to untouched past `falloff`.
//!
//! The curves are smooth (Catmull-Rom through their points, closed or
//! open). On the CPU each curve is sampled to [`WARP_SAMPLES`] points at the
//! same curve parameters on both sides, so a source sample and its
//! destination sample are one correspondence. Per output pixel the shader
//! takes every destination segment, finds the nearest point on it, and
//! weighs that point's displacement (source minus destination, interpolated
//! along the segment) by the segment's length over the squared distance;
//! a constant weight for "no displacement" sets the falloff. On a
//! destination curve the nearest segment dominates, so the pixel samples
//! exactly its source point. The input is then sampled there
//! (`Emit::input_at`, as Transform and GridWarp do): it fuses, no pass.
//!
//! Every point is a uniform (an animatable channel in the app). Only the
//! NUMBER of curve pairs is structure: it sets the uniform span and the
//! shader's loop; the point count of a curve never is, because a curve is
//! always packed as the same number of samples.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x3a59_0000;

/// The most curve pairs one node carries.
pub const WARP_PAIRS: usize = 8;
/// Samples per curve on the GPU, whatever its point count.
pub const WARP_SAMPLES: usize = 24;
/// Header vec4s before the samples.
const HEADER: usize = 2;
/// Squared pixels added to every distance: what keeps a pixel ON a curve
/// finite, and small enough that it lands on its source point.
const EPS: f32 = 1e-2;

/// One pair: a source curve, a destination curve, the same number of
/// points in each (normalised over the node's format, y down).
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct WarpPair {
    pub src: Vec<[f32; 2]>,
    pub dst: Vec<[f32; 2]>,
    /// The curve closes on itself (a loop round a shape).
    pub closed: bool,
}

/// A [`crate::graph::NodeKind::Warp`]'s curves and controls.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct WarpParams {
    pub pairs: Vec<WarpPair>,
    /// How far from the curves the warp reaches before it fades, in
    /// full-resolution pixels: at this distance from a long curve it has
    /// fallen to about half.
    pub falloff: f32,
    /// 0 = untouched, 1 = the full warp.
    pub amount: f32,
}

impl Default for WarpParams {
    fn default() -> Self {
        Self { pairs: Vec::new(), falloff: 200.0, amount: 1.0 }
    }
}

impl WarpPair {
    /// The pair's points as one curve's worth: both sides, in step.
    pub fn len(&self) -> usize {
        self.src.len().min(self.dst.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `WARP_SAMPLES` points along one side (`dst` or not), in the units
    /// its points are in. Catmull-Rom through the points, the ends
    /// extended by reflection; closed wraps.
    pub fn samples(&self, dst: bool) -> Vec<[f32; 2]> {
        let pts = if dst { &self.dst[..self.len()] } else { &self.src[..self.len()] };
        sample_curve(pts, self.closed)
    }
}

/// `WARP_SAMPLES` points along a Catmull-Rom curve through `pts`.
pub fn sample_curve(pts: &[[f32; 2]], closed: bool) -> Vec<[f32; 2]> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![pts[0]; WARP_SAMPLES];
    }
    let closed = closed && n >= 3;
    let segs = if closed { n } else { n - 1 };
    let at = |i: isize| -> [f32; 2] {
        if closed {
            pts[i.rem_euclid(n as isize) as usize]
        } else if i < 0 {
            // Reflect the first point through its neighbour's opposite.
            let (a, b) = (pts[0], pts[1]);
            [2.0 * a[0] - b[0], 2.0 * a[1] - b[1]]
        } else if i as usize >= n {
            let (a, b) = (pts[n - 1], pts[n - 2]);
            [2.0 * a[0] - b[0], 2.0 * a[1] - b[1]]
        } else {
            pts[i as usize]
        }
    };
    (0..WARP_SAMPLES)
        .map(|k| {
            let t = k as f32 / (WARP_SAMPLES - 1) as f32 * segs as f32;
            let i = (t.floor() as usize).min(segs - 1);
            let f = t - i as f32;
            let i = i as isize;
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            let (f2, f3) = (f * f, f * f * f);
            let w = [
                0.5 * (-f3 + 2.0 * f2 - f),
                0.5 * (3.0 * f3 - 5.0 * f2 + 2.0),
                0.5 * (-3.0 * f3 + 4.0 * f2 + f),
                0.5 * (f3 - f2),
            ];
            let c = |l: usize| w[0] * p0[l] + w[1] * p1[l] + w[2] * p2[l] + w[3] * p3[l];
            [c(0), c(1)]
        })
        .collect()
}

impl WarpParams {
    /// The pairs that reach the GPU: the first [`WARP_PAIRS`] with a point.
    pub fn live_pairs(&self) -> impl Iterator<Item = &WarpPair> {
        self.pairs.iter().filter(|p| !p.is_empty()).take(WARP_PAIRS)
    }

    /// The uniform span.
    pub fn slots(&self) -> usize {
        HEADER + self.live_pairs().count() * WARP_SAMPLES
    }

    /// Every (source px, destination px) sample, pair by pair: what the
    /// packer writes and the CPU mirror reads.
    fn sample_px(&self, fmt: [f32; 2]) -> Vec<Vec<[f32; 4]>> {
        self.live_pairs()
            .map(|p| {
                let (s, d) = (p.samples(false), p.samples(true));
                s.iter()
                    .zip(&d)
                    .map(|(s, d)| [s[0] * fmt[0], s[1] * fmt[1], d[0] * fmt[0], d[1] * fmt[1]])
                    .collect()
            })
            .collect()
    }

    /// Where output pixel `p` samples the source (pixels of a `fmt`
    /// format, `falloff` already in those pixels). The CPU mirror of
    /// `warp_px`.
    pub fn sample_at(&self, p: [f32; 2], fmt: [f32; 2], falloff: f32) -> [f32; 2] {
        let mut wsum = std::f32::consts::PI / falloff.max(1.0);
        let mut d = [0.0f32; 2];
        for pair in self.sample_px(fmt) {
            for k in 0..WARP_SAMPLES - 1 {
                let (a, b) = (pair[k], pair[k + 1]);
                let seg = [b[2] - a[2], b[3] - a[3]];
                let l2 = seg[0] * seg[0] + seg[1] * seg[1];
                let u = (((p[0] - a[2]) * seg[0] + (p[1] - a[3]) * seg[1]) / l2.max(1e-6)).clamp(0.0, 1.0);
                let q = [a[2] + seg[0] * u, a[3] + seg[1] * u];
                let dist2 = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2);
                let w = (l2.sqrt() + 1.0) / (dist2 + EPS);
                for l in 0..2 {
                    let da = a[l] - a[2 + l];
                    let db = b[l] - b[2 + l];
                    d[l] += w * (da + (db - da) * u);
                }
                wsum += w;
            }
        }
        [p[0] + d[0] / wsum * self.amount, p[1] + d[1] / wsum * self.amount]
    }

    /// The span with the node's format and the proxy in it: the codegen's
    /// central packer knows both, `pack` does not.
    pub fn pack_fmt(&self, fmt: [f32; 2], proxy: f32, v: &mut [f32]) {
        v[0] = self.live_pairs().count() as f32;
        v[1] = (self.falloff * proxy).max(1.0);
        v[2] = fmt[0];
        v[3] = fmt[1];
        v[4] = self.amount;
        for (pi, pair) in self.sample_px(fmt).iter().enumerate() {
            for (k, s) in pair.iter().enumerate() {
                let o = (HEADER + pi * WARP_SAMPLES + k) * 4;
                if o + 4 <= v.len() {
                    v[o..o + 4].copy_from_slice(s);
                }
            }
        }
    }
}

impl NodeDef for WarpParams {
    const NAME: &'static str = "Warp";
    /// The source, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// The HEADER only: the samples follow, and the codegen reads the
    /// whole span from [`WarpParams::slots`].
    const VEC4_SLOTS: usize = HEADER;

    /// The number of curve pairs: the span and the loop. Points are
    /// uniforms.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE + self.live_pairs().count() as u64
    }

    /// Without the format there is nothing to pack in pixels; see
    /// [`WarpParams::pack_fmt`].
    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt([1.0, 1.0], 1.0, v);
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(0) {
            return "vec4f(0.0)".into();
        }
        let pairs = self.live_pairs().count();
        if pairs == 0 {
            return e.input(0);
        }
        let name = wgsl(e, pairs);
        let slot = e.slot();
        let pos = e.pos().to_string();
        let q = e.fresh();
        e.line(format_args!("let {q} = {name}({pos}, {slot}u);"));
        let v = e.input_at(0, &q);
        if e.has_input(1) {
            let orig = e.input(0);
            e.masked(Self::MASK_INPUT, &orig, v)
        } else {
            v
        }
    }
}

/// Emit `warp_px{pairs}` once per shader; returns its name.
fn wgsl(e: &mut Emit, pairs: usize) -> String {
    let name = format!("warp_px{pairs}");
    let (_, fns) = e.color_helpers();
    if fns.contains_key(&name) {
        return name;
    }
    let samples = WARP_SAMPLES;
    let segs = samples - 1;
    let pi = std::f32::consts::PI;
    fns.insert(
        name.clone(),
        format!(
            "// Warp over {pairs} curve pair(s): P.v[slot + {HEADER} + pair * {samples} + k] =
// (source.xy, destination.xy) in pixels. Each destination segment pulls
// toward its own displacement by length / distance^2; a constant term
// fades the warp out past the falloff.
fn {name}(p: vec2f, slot: u32) -> vec2f {{
    let h = P.v[slot];
    var wsum = {pi:?} / max(h.y, 1.0);
    var d = vec2f(0.0);
    for (var pr = 0u; pr < {pairs}u; pr = pr + 1u) {{
        let base = slot + {HEADER}u + pr * {samples}u;
        for (var k = 0u; k < {segs}u; k = k + 1u) {{
            let a = P.v[base + k];
            let b = P.v[base + k + 1u];
            let seg = b.zw - a.zw;
            let l2 = dot(seg, seg);
            let u = clamp(dot(p - a.zw, seg) / max(l2, 1e-6), 0.0, 1.0);
            let q = a.zw + seg * u;
            let r = p - q;
            let w = (sqrt(l2) + 1.0) / (dot(r, r) + {EPS:?});
            d = d + w * mix(a.xy - a.zw, b.xy - b.zw, u);
            wsum = wsum + w;
        }}
    }}
    return p + d / wsum * P.v[slot + 1u].x;
}}
"
        ),
    );
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    const FMT: [f32; 2] = [400.0, 200.0];

    fn line_pair(dy: f32) -> WarpPair {
        let src = vec![[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]];
        let dst = src.iter().map(|p| [p[0], p[1] + dy]).collect();
        WarpPair { src, dst, closed: false }
    }

    #[test]
    fn no_curves_is_the_identity() {
        let p = WarpParams::default();
        assert_eq!(p.sample_at([120.0, 33.0], FMT, 200.0), [120.0, 33.0]);
    }

    #[test]
    fn identical_curves_do_nothing() {
        let p = WarpParams { pairs: vec![line_pair(0.0)], ..Default::default() };
        for q in [[10.0, 10.0], [200.0, 100.0], [300.0, 180.0]] {
            let s = p.sample_at(q, FMT, 200.0);
            assert!((s[0] - q[0]).abs() < 1e-3 && (s[1] - q[1]).abs() < 1e-3, "{q:?} -> {s:?}");
        }
    }

    /// A pixel on the destination curve samples (nearly exactly) its
    /// source curve, and far away the warp has faded.
    #[test]
    fn the_destination_curve_samples_the_source_and_fades_out() {
        let p = WarpParams { pairs: vec![line_pair(0.1)], ..Default::default() };
        // The middle of the destination line is (200, 120); its source is
        // (200, 100).
        let s = p.sample_at([200.0, 120.0], FMT, 50.0);
        assert!((s[0] - 200.0).abs() < 0.2 && (s[1] - 100.0).abs() < 0.2, "{s:?}");
        let far = p.sample_at([200.0, 199.0], FMT, 5.0);
        assert!((far[1] - 199.0).abs() < 3.0, "far away the warp should fade: {far:?}");
        let mid = p.sample_at([200.0, 140.0], FMT, 50.0);
        assert!(mid[1] < 140.0 && mid[1] > 120.0, "{mid:?}");
    }

    #[test]
    fn amount_scales_the_warp() {
        let mut p = WarpParams { pairs: vec![line_pair(0.1)], ..Default::default() };
        p.amount = 0.0;
        assert_eq!(p.sample_at([200.0, 120.0], FMT, 50.0), [200.0, 120.0]);
    }

    #[test]
    fn samples_pass_through_the_points() {
        let pts = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let s = sample_curve(&pts, false);
        assert_eq!(s.len(), WARP_SAMPLES);
        assert!((s[0][0] - 0.0).abs() < 1e-6 && (s[WARP_SAMPLES - 1][1] - 1.0).abs() < 1e-6);
        let c = sample_curve(&pts, true);
        let (a, b) = (c[0], c[WARP_SAMPLES - 1]);
        assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6, "a closed curve ends where it starts");
    }

    #[test]
    fn only_the_pair_count_is_topology() {
        let a = WarpParams { pairs: vec![line_pair(0.0)], ..Default::default() };
        let mut b = a.clone();
        b.pairs[0].dst[1] = [0.4, 0.9];
        b.pairs[0].src.push([0.9, 0.9]);
        b.pairs[0].dst.push([0.9, 0.9]);
        b.falloff = 12.0;
        assert_eq!(a.topo_tag(), b.topo_tag());
        let mut c = a.clone();
        c.pairs.push(line_pair(0.1));
        assert_ne!(a.topo_tag(), c.topo_tag());
        // An empty pair is not packed and not structure.
        let mut d = a.clone();
        d.pairs.push(WarpPair::default());
        assert_eq!(a.topo_tag(), d.topo_tag());
        let full = WarpParams { pairs: vec![line_pair(0.0); WARP_PAIRS + 3], ..Default::default() };
        assert!(full.slots() * 16 <= 16384 / 4, "{} bytes", full.slots() * 16);
    }

    #[test]
    fn the_generated_shader_validates() {
        use crate::graph::{Graph, NodeKind};
        for masked in [false, true] {
            let mut g = Graph::default();
            g.nodes.insert(0, NodeKind::read(0, ""));
            g.nodes.insert(1, NodeKind::read(1, ""));
            let p = WarpParams { pairs: vec![line_pair(0.1), line_pair(-0.05)], ..Default::default() };
            g.nodes.insert(2, NodeKind::Warp(p));
            g.edges.insert((2, 0), 0);
            if masked {
                g.edges.insert((2, 1), 1);
            }
            g.view = Some(2);
            let src = crate::codegen::generate(&g).main.wgsl;
            assert!(src.contains("fn warp_px2("), "{src}");
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("validate: {e:?}"));
        }
    }
}
